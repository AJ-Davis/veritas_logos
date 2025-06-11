"""
Test suite for the billing system.

This module tests subscription management, payment processing, 
usage tracking, and customer management functionality.
"""

import pytest
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import the models and functions to test
from src.api.billing import (
    Base, Customer, Subscription, Payment, UsageRecord,
    SubscriptionTier, SubscriptionStatus, PaymentStatus, UsageType,
    SUBSCRIPTION_PLANS, create_customer, get_customer_by_user,
    create_subscription, get_active_subscription, record_usage,
    get_usage_stats, check_usage_limit, enforce_usage_limit,
    CustomerCreate, SubscriptionCreate, UsageRecordCreate
)
from src.api.auth import User
from src.api.main import app

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_billing.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture
def db():
    """Create test database session."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture
def test_user():
    """Create test user."""
    return User(
        id="test-user-id",
        email="test@example.com",
        full_name="Test User",
        is_active=True
    )

@pytest.fixture
def test_customer_data():
    """Create test customer data."""
    return CustomerCreate(
        email="test@example.com",
        full_name="Test User"
    )

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestSubscriptionPlans:
    """Test subscription plan configuration."""
    
    def test_subscription_plans_exist(self):
        """Test that all subscription plans are defined."""
        assert SubscriptionTier.FREE in SUBSCRIPTION_PLANS
        assert SubscriptionTier.STARTER in SUBSCRIPTION_PLANS
        assert SubscriptionTier.PROFESSIONAL in SUBSCRIPTION_PLANS
        assert SubscriptionTier.ENTERPRISE in SUBSCRIPTION_PLANS
    
    def test_plan_structure(self):
        """Test that each plan has required fields."""
        for tier, plan in SUBSCRIPTION_PLANS.items():
            assert hasattr(plan, 'tier')
            assert hasattr(plan, 'name')
            assert hasattr(plan, 'price_monthly')
            assert hasattr(plan, 'price_yearly')
            assert hasattr(plan, 'features')
            assert hasattr(plan, 'limits')
            assert plan.tier == tier
    
    def test_free_plan_pricing(self):
        """Test that free plan has zero pricing."""
        free_plan = SUBSCRIPTION_PLANS[SubscriptionTier.FREE]
        assert free_plan.price_monthly == 0
        assert free_plan.price_yearly == 0


class TestCustomerManagement:
    """Test customer management functionality."""
    
    @patch('src.api.billing.stripe.Customer.create')
    def test_create_customer(self, mock_stripe_create, db, test_user, test_customer_data):
        """Test customer creation."""
        # Mock Stripe customer creation
        mock_stripe_create.return_value = MagicMock(id="cus_test123")
        
        customer = create_customer(db, test_user, test_customer_data)
        
        assert customer.user_id == test_user.id
        assert customer.email == test_customer_data.email
        assert customer.stripe_customer_id == "cus_test123"
        mock_stripe_create.assert_called_once()
    
    def test_get_customer_by_user(self, db, test_user):
        """Test getting customer by user."""
        # Initially no customer should exist
        customer = get_customer_by_user(db, test_user)
        assert customer is None
        
        # Create customer
        test_customer = Customer(
            user_id=test_user.id,
            stripe_customer_id="cus_test123",
            email=test_user.email
        )
        db.add(test_customer)
        db.commit()
        
        # Now customer should be found
        customer = get_customer_by_user(db, test_user)
        assert customer is not None
        assert customer.user_id == test_user.id


class TestSubscriptionManagement:
    """Test subscription management functionality."""
    
    @patch('src.api.billing.stripe.Subscription.create')
    def test_create_subscription(self, mock_stripe_create, db, test_user):
        """Test subscription creation."""
        # Create customer first
        customer = Customer(
            user_id=test_user.id,
            stripe_customer_id="cus_test123",
            email=test_user.email
        )
        db.add(customer)
        db.commit()
        
        # Mock Stripe subscription creation
        mock_stripe_create.return_value = MagicMock(
            id="sub_test123",
            current_period_start=1640995200,  # 2022-01-01
            current_period_end=1643673600     # 2022-02-01
        )
        
        subscription_data = SubscriptionCreate(
            tier=SubscriptionTier.STARTER,
            billing_interval="month"
        )
        
        subscription = create_subscription(db, customer, subscription_data)
        
        assert subscription.customer_id == customer.id
        assert subscription.tier == SubscriptionTier.STARTER
        assert subscription.status == SubscriptionStatus.ACTIVE
        mock_stripe_create.assert_called_once()
    
    def test_get_active_subscription(self, db, test_user):
        """Test getting active subscription."""
        # Create customer
        customer = Customer(
            user_id=test_user.id,
            stripe_customer_id="cus_test123",
            email=test_user.email
        )
        db.add(customer)
        db.commit()
        
        # Initially no subscription
        subscription = get_active_subscription(db, customer)
        assert subscription is None
        
        # Create active subscription
        test_subscription = Subscription(
            customer_id=customer.id,
            tier=SubscriptionTier.STARTER,
            status=SubscriptionStatus.ACTIVE
        )
        db.add(test_subscription)
        db.commit()
        
        # Should find active subscription
        subscription = get_active_subscription(db, customer)
        assert subscription is not None
        assert subscription.tier == SubscriptionTier.STARTER


class TestUsageTracking:
    """Test usage tracking functionality."""
    
    def test_record_usage(self, db, test_user):
        """Test recording usage."""
        # Create customer
        customer = Customer(
            user_id=test_user.id,
            stripe_customer_id="cus_test123",
            email=test_user.email
        )
        db.add(customer)
        db.commit()
        
        usage_data = UsageRecordCreate(
            usage_type=UsageType.DOCUMENT_VERIFICATION,
            quantity=1
        )
        
        usage_record = record_usage(db, customer, usage_data)
        
        assert usage_record.customer_id == customer.id
        assert usage_record.usage_type == UsageType.DOCUMENT_VERIFICATION
        assert usage_record.quantity == 1
    
    def test_get_usage_stats(self, db, test_user):
        """Test getting usage statistics."""
        # Create customer with subscription
        customer = Customer(
            user_id=test_user.id,
            stripe_customer_id="cus_test123",
            email=test_user.email
        )
        subscription = Subscription(
            customer_id=customer.id,
            tier=SubscriptionTier.STARTER,
            status=SubscriptionStatus.ACTIVE
        )
        db.add(customer)
        db.add(subscription)
        db.commit()
        
        # Record some usage
        now = datetime.utcnow()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        period_end = (period_start + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)
        
        usage_record = UsageRecord(
            customer_id=customer.id,
            usage_type=UsageType.DOCUMENT_VERIFICATION,
            quantity=5,
            billing_period_start=period_start,
            billing_period_end=period_end
        )
        db.add(usage_record)
        db.commit()
        
        stats = get_usage_stats(db, customer)
        
        assert stats.usage_by_type[UsageType.DOCUMENT_VERIFICATION] == 5
        assert stats.total_usage == 5
    
    def test_check_usage_limit(self, db, test_user):
        """Test checking usage limits."""
        # Create customer with subscription
        customer = Customer(
            user_id=test_user.id,
            stripe_customer_id="cus_test123",
            email=test_user.email
        )
        subscription = Subscription(
            customer_id=customer.id,
            tier=SubscriptionTier.FREE,  # Free plan has limits
            status=SubscriptionStatus.ACTIVE
        )
        db.add(customer)
        db.add(subscription)
        db.commit()
        
        # Should be allowed initially
        allowed = check_usage_limit(db, customer, UsageType.DOCUMENT_VERIFICATION, 1)
        assert allowed == True


if __name__ == "__main__":
    pytest.main([__file__]) 
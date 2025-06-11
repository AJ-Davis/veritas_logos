"""
Stripe billing integration for the Veritas Logos API.

This module implements comprehensive billing functionality including:
- Subscription management with tiered plans
- Payment processing and webhook handling
- Usage tracking and billing enforcement
- Customer management and invoicing
"""

import os
import stripe
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum

from fastapi import HTTPException, status, Depends
from pydantic import BaseModel, Field, validator
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Enum as SQLEnum, Integer, Float, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.postgresql import UUID

from .auth import get_db, get_current_user, User

# Configure Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

# Database setup (reusing from auth.py)
Base = declarative_base()


class SubscriptionTier(str, Enum):
    """Subscription tier options."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class SubscriptionStatus(str, Enum):
    """Subscription status options."""
    ACTIVE = "active"
    CANCELLED = "cancelled"
    PAST_DUE = "past_due"
    TRIALING = "trialing"
    INCOMPLETE = "incomplete"
    INCOMPLETE_EXPIRED = "incomplete_expired"
    UNPAID = "unpaid"


class PaymentStatus(str, Enum):
    """Payment status options."""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"


class UsageType(str, Enum):
    """Usage tracking types."""
    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_VERIFICATION = "document_verification"
    PAGE_ANALYSIS = "page_analysis" 
    API_CALL = "api_call"
    STORAGE_MB = "storage_mb"
    EXPORT_PDF = "export_pdf"
    EXPORT_DOCX = "export_docx"
    DASHBOARD_VIEW = "dashboard_view"


# SQLAlchemy Models
class Customer(Base):
    """Customer billing information."""
    __tablename__ = "customers"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), unique=True, nullable=False)
    stripe_customer_id = Column(String, unique=True, nullable=False)
    email = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="customer")
    subscriptions = relationship("Subscription", back_populates="customer")
    payments = relationship("Payment", back_populates="customer")
    usage_records = relationship("UsageRecord", back_populates="customer")


class Subscription(Base):
    """User subscription information."""
    __tablename__ = "subscriptions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    customer_id = Column(String, ForeignKey("customers.id"), nullable=False)
    stripe_subscription_id = Column(String, unique=True, nullable=True)
    tier = Column(SQLEnum(SubscriptionTier), nullable=False, default=SubscriptionTier.FREE)
    status = Column(SQLEnum(SubscriptionStatus), nullable=False, default=SubscriptionStatus.ACTIVE)
    current_period_start = Column(DateTime)
    current_period_end = Column(DateTime)
    trial_start = Column(DateTime)
    trial_end = Column(DateTime)
    cancel_at_period_end = Column(Boolean, default=False)
    cancelled_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    customer = relationship("Customer", back_populates="subscriptions")


class Payment(Base):
    """Payment transaction records."""
    __tablename__ = "payments"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    customer_id = Column(String, ForeignKey("customers.id"), nullable=False)
    stripe_payment_intent_id = Column(String, unique=True, nullable=True)
    stripe_invoice_id = Column(String, nullable=True)
    amount = Column(Integer, nullable=False)  # Amount in cents
    currency = Column(String, default="usd")
    status = Column(SQLEnum(PaymentStatus), nullable=False, default=PaymentStatus.PENDING)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    customer = relationship("Customer", back_populates="payments")


class UsageRecord(Base):
    """Usage tracking for billing enforcement."""
    __tablename__ = "usage_records"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    customer_id = Column(String, ForeignKey("customers.id"), nullable=False)
    usage_type = Column(SQLEnum(UsageType), nullable=False)
    quantity = Column(Integer, nullable=False, default=1)
    context_metadata = Column(Text)  # JSON string for additional context
    recorded_at = Column(DateTime, default=datetime.utcnow)
    billing_period_start = Column(DateTime, nullable=False)
    billing_period_end = Column(DateTime, nullable=False)
    
    # Relationships
    customer = relationship("Customer", back_populates="usage_records")


# Pydantic Models
class SubscriptionPlan(BaseModel):
    """Subscription plan configuration."""
    tier: SubscriptionTier
    name: str
    price_monthly: int  # In cents
    price_yearly: int   # In cents
    features: List[str]
    limits: Dict[str, int]  # Usage limits
    stripe_price_id_monthly: Optional[str] = None
    stripe_price_id_yearly: Optional[str] = None


class CustomerCreate(BaseModel):
    """Customer creation request."""
    email: str
    full_name: Optional[str]
    metadata: Optional[Dict[str, str]] = {}


class CustomerResponse(BaseModel):
    """Customer response model."""
    id: str
    user_id: str
    stripe_customer_id: str
    email: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class SubscriptionCreate(BaseModel):
    """Subscription creation request."""
    tier: SubscriptionTier
    billing_interval: str = Field(default="month", pattern="^(month|year)$")
    trial_period_days: Optional[int] = None


class SubscriptionResponse(BaseModel):
    """Subscription response model."""
    id: str
    tier: SubscriptionTier
    status: SubscriptionStatus
    current_period_start: Optional[datetime]
    current_period_end: Optional[datetime]
    trial_end: Optional[datetime]
    cancel_at_period_end: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class PaymentIntentCreate(BaseModel):
    """Payment intent creation request."""
    amount: int  # In cents
    currency: str = "usd"
    description: Optional[str]
    metadata: Optional[Dict[str, str]] = {}


class PaymentResponse(BaseModel):
    """Payment response model."""
    id: str
    amount: int
    currency: str
    status: PaymentStatus
    description: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class UsageRecordCreate(BaseModel):
    """Usage record creation request."""
    usage_type: UsageType
    quantity: int = 1
    metadata: Optional[Dict[str, Any]] = {}


class UsageStats(BaseModel):
    """Usage statistics for a billing period."""
    billing_period_start: datetime
    billing_period_end: datetime
    usage_by_type: Dict[UsageType, int]
    total_usage: int
    limits: Dict[UsageType, int]
    usage_remaining: Dict[UsageType, int]


# Subscription Plans Configuration
SUBSCRIPTION_PLANS: Dict[SubscriptionTier, SubscriptionPlan] = {
    SubscriptionTier.FREE: SubscriptionPlan(
        tier=SubscriptionTier.FREE,
        name="Free",
        price_monthly=0,
        price_yearly=0,
        features=[
            "5 document verifications per month",
            "Basic verification passes",
            "PDF export",
            "Community support"
        ],
        limits={
            "document_upload": 10,
            "document_verification": 5,
            "page_analysis": 50,
            "api_call": 100,
            "storage_mb": 100,
            "export_pdf": 5,
            "export_docx": 0,
            "dashboard_view": 10
        }
    ),
    SubscriptionTier.STARTER: SubscriptionPlan(
        tier=SubscriptionTier.STARTER,
        name="Starter",
        price_monthly=2900,  # $29/month
        price_yearly=29000,  # $290/year (2 months free)
        features=[
            "100 document verifications per month",
            "All verification passes",
            "PDF & DOCX export",
            "Dashboard analytics",
            "Email support"
        ],
        limits={
            "document_upload": 200,
            "document_verification": 100,
            "page_analysis": 1000,
            "api_call": 2000,
            "storage_mb": 1000,
            "export_pdf": 100,
            "export_docx": 100,
            "dashboard_view": 100
        },
        stripe_price_id_monthly=os.getenv("STRIPE_STARTER_MONTHLY_PRICE_ID"),
        stripe_price_id_yearly=os.getenv("STRIPE_STARTER_YEARLY_PRICE_ID")
    ),
    SubscriptionTier.PROFESSIONAL: SubscriptionPlan(
        tier=SubscriptionTier.PROFESSIONAL,
        name="Professional",
        price_monthly=9900,  # $99/month
        price_yearly=99000,  # $990/year (2 months free)
        features=[
            "Unlimited document verifications",
            "All verification passes",
            "Advanced debate view",
            "API access",
            "Priority support",
            "Custom integrations"
        ],
        limits={
            "document_upload": -1,  # Unlimited
            "document_verification": -1,  # Unlimited
            "page_analysis": -1,
            "api_call": 10000,
            "storage_mb": 10000,
            "export_pdf": -1,
            "export_docx": -1,
            "dashboard_view": -1
        },
        stripe_price_id_monthly=os.getenv("STRIPE_PROFESSIONAL_MONTHLY_PRICE_ID"),
        stripe_price_id_yearly=os.getenv("STRIPE_PROFESSIONAL_YEARLY_PRICE_ID")
    ),
    SubscriptionTier.ENTERPRISE: SubscriptionPlan(
        tier=SubscriptionTier.ENTERPRISE,
        name="Enterprise",
        price_monthly=29900,  # $299/month
        price_yearly=299000,  # $2990/year (2 months free)
        features=[
            "Unlimited everything",
            "Dedicated instance",
            "Custom verification chains",
            "SLA guarantee",
            "24/7 support",
            "On-premise deployment option"
        ],
        limits={
            "document_upload": -1,
            "document_verification": -1,
            "page_analysis": -1,
            "api_call": -1,
            "storage_mb": -1,
            "export_pdf": -1,
            "export_docx": -1,
            "dashboard_view": -1
        },
        stripe_price_id_monthly=os.getenv("STRIPE_ENTERPRISE_MONTHLY_PRICE_ID"),
        stripe_price_id_yearly=os.getenv("STRIPE_ENTERPRISE_YEARLY_PRICE_ID")
    )
}


# Business Logic Functions
def create_customer(db: Session, user: User, customer_data: CustomerCreate) -> Customer:
    """Create a new customer in Stripe and database."""
    try:
        # Create customer in Stripe
        stripe_customer = stripe.Customer.create(
            email=customer_data.email,
            name=customer_data.full_name,
            metadata={
                "user_id": user.id,
                "username": user.username,
                **customer_data.metadata
            }
        )
        
        # Create customer in database
        customer = Customer(
            user_id=user.id,
            stripe_customer_id=stripe_customer.id,
            email=customer_data.email
        )
        
        db.add(customer)
        db.commit()
        db.refresh(customer)
        
        return customer
        
    except stripe.error.StripeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Stripe error: {str(e)}"
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating customer: {str(e)}"
        )


def get_customer_by_user(db: Session, user: User) -> Optional[Customer]:
    """Get customer by user ID."""
    return db.query(Customer).filter(Customer.user_id == user.id).first()


def create_subscription(db: Session, customer: Customer, subscription_data: SubscriptionCreate) -> Subscription:
    """Create a new subscription."""
    plan = SUBSCRIPTION_PLANS[subscription_data.tier]
    
    try:
        if subscription_data.tier == SubscriptionTier.FREE:
            # Free tier doesn't need Stripe subscription
            subscription = Subscription(
                customer_id=customer.id,
                tier=subscription_data.tier,
                status=SubscriptionStatus.ACTIVE,
                current_period_start=datetime.utcnow(),
                current_period_end=datetime.utcnow() + timedelta(days=30)
            )
        else:
            # Get appropriate price ID
            price_id = (plan.stripe_price_id_yearly if subscription_data.billing_interval == "year" 
                       else plan.stripe_price_id_monthly)
            
            if not price_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Price ID not configured for {subscription_data.tier} {subscription_data.billing_interval}"
                )
            
            # Create Stripe subscription
            create_params = {
                "customer": customer.stripe_customer_id,
                "items": [{"price": price_id}],
                "metadata": {
                    "user_id": customer.user_id,
                    "tier": subscription_data.tier.value
                }
            }
            
            if subscription_data.trial_period_days:
                create_params["trial_period_days"] = subscription_data.trial_period_days
            
            stripe_subscription = stripe.Subscription.create(**create_params)
            
            # Create subscription in database
            subscription = Subscription(
                customer_id=customer.id,
                stripe_subscription_id=stripe_subscription.id,
                tier=subscription_data.tier,
                status=SubscriptionStatus(stripe_subscription.status),
                current_period_start=datetime.fromtimestamp(stripe_subscription.current_period_start, tz=timezone.utc),
                current_period_end=datetime.fromtimestamp(stripe_subscription.current_period_end, tz=timezone.utc),
                trial_start=datetime.fromtimestamp(stripe_subscription.trial_start, tz=timezone.utc) if stripe_subscription.trial_start else None,
                trial_end=datetime.fromtimestamp(stripe_subscription.trial_end, tz=timezone.utc) if stripe_subscription.trial_end else None
            )
        
        db.add(subscription)
        db.commit()
        db.refresh(subscription)
        
        return subscription
        
    except stripe.error.StripeError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Stripe error: {str(e)}"
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating subscription: {str(e)}"
        )


def get_active_subscription(db: Session, customer: Customer) -> Optional[Subscription]:
    """Get the active subscription for a customer."""
    return (db.query(Subscription)
            .filter(
                Subscription.customer_id == customer.id,
                Subscription.status.in_([SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING])
            )
            .order_by(Subscription.created_at.desc())
            .first())


def create_payment_intent(db: Session, customer: Customer, payment_data: PaymentIntentCreate) -> Dict[str, Any]:
    """Create a payment intent for one-time payments."""
    try:
        # Create payment intent in Stripe
        intent = stripe.PaymentIntent.create(
            amount=payment_data.amount,
            currency=payment_data.currency,
            customer=customer.stripe_customer_id,
            description=payment_data.description,
            metadata=payment_data.metadata,
            automatic_payment_methods={'enabled': True}
        )
        
        # Create payment record in database
        payment = Payment(
            customer_id=customer.id,
            stripe_payment_intent_id=intent.id,
            amount=payment_data.amount,
            currency=payment_data.currency,
            status=PaymentStatus.PENDING,
            description=payment_data.description
        )
        
        db.add(payment)
        db.commit()
        
        return {
            "client_secret": intent.client_secret,
            "payment_intent_id": intent.id
        }
        
    except stripe.error.StripeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Stripe error: {str(e)}"
        )


def record_usage(db: Session, customer: Customer, usage_data: UsageRecordCreate) -> UsageRecord:
    """Record usage for billing enforcement."""
    # Get current billing period
    subscription = get_active_subscription(db, customer)
    if subscription:
        period_start = subscription.current_period_start
        period_end = subscription.current_period_end
    else:
        # For free users, use monthly periods
        now = datetime.utcnow()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if now.month == 12:
            period_end = period_start.replace(year=now.year + 1, month=1)
        else:
            period_end = period_start.replace(month=now.month + 1)
    
    usage_record = UsageRecord(
        customer_id=customer.id,
        usage_type=usage_data.usage_type,
        quantity=usage_data.quantity,
        context_metadata=str(usage_data.metadata) if usage_data.metadata else None,
        billing_period_start=period_start,
        billing_period_end=period_end
    )
    
    db.add(usage_record)
    db.commit()
    db.refresh(usage_record)
    
    return usage_record


def get_usage_stats(db: Session, customer: Customer) -> UsageStats:
    """Get usage statistics for current billing period."""
    subscription = get_active_subscription(db, customer)
    tier = subscription.tier if subscription else SubscriptionTier.FREE
    plan = SUBSCRIPTION_PLANS[tier]
    
    if subscription:
        period_start = subscription.current_period_start
        period_end = subscription.current_period_end
    else:
        # For free users, use monthly periods
        now = datetime.utcnow()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if now.month == 12:
            period_end = period_start.replace(year=now.year + 1, month=1)
        else:
            period_end = period_start.replace(month=now.month + 1)
    
    # Get usage records for current period
    usage_records = (db.query(UsageRecord)
                    .filter(
                        UsageRecord.customer_id == customer.id,
                        UsageRecord.billing_period_start == period_start
                    )
                    .all())
    
    # Aggregate usage by type
    usage_by_type = {}
    for usage_type in UsageType:
        usage_by_type[usage_type] = sum(
            record.quantity for record in usage_records 
            if record.usage_type == usage_type
        )
    
    # Calculate remaining usage
    usage_remaining = {}
    for usage_type, limit in plan.limits.items():
        usage_type_enum = UsageType(usage_type)
        current_usage = usage_by_type.get(usage_type_enum, 0)
        if limit == -1:  # Unlimited
            usage_remaining[usage_type_enum] = -1
        else:
            usage_remaining[usage_type_enum] = max(0, limit - current_usage)
    
    return UsageStats(
        billing_period_start=period_start,
        billing_period_end=period_end,
        usage_by_type=usage_by_type,
        total_usage=sum(usage_by_type.values()),
        limits={UsageType(k): v for k, v in plan.limits.items()},
        usage_remaining=usage_remaining
    )


def check_usage_limit(db: Session, customer: Customer, usage_type: UsageType, quantity: int = 1) -> bool:
    """Check if usage is within limits."""
    stats = get_usage_stats(db, customer)
    remaining = stats.usage_remaining.get(usage_type, 0)
    
    if remaining == -1:  # Unlimited
        return True
    
    return remaining >= quantity


def enforce_usage_limit(db: Session, customer: Customer, usage_type: UsageType, quantity: int = 1):
    """Enforce usage limits by raising exception if exceeded."""
    if not check_usage_limit(db, customer, usage_type, quantity):
        subscription = get_active_subscription(db, customer)
        tier = subscription.tier if subscription else SubscriptionTier.FREE
        
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Usage limit exceeded for {usage_type.value}. Please upgrade your {tier.value} plan."
        ) 
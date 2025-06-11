"""
Billing API routes for the Veritas Logos system.

This module implements all billing-related endpoints including subscription
management, payment processing, usage tracking, and customer management.
"""

import stripe
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from fastapi import APIRouter, Depends, HTTPException, status, Request, Header
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from src.api.auth import get_db, get_current_user, get_current_admin_user, User
from src.api.billing import (
    # Models
    Customer, Subscription, Payment, UsageRecord,
    SubscriptionTier, SubscriptionStatus, PaymentStatus, UsageType,
    
    # Pydantic Models
    SubscriptionPlan, CustomerCreate, CustomerResponse, SubscriptionCreate, 
    SubscriptionResponse, PaymentIntentCreate, PaymentResponse, 
    UsageRecordCreate, UsageStats,
    
    # Business Logic
    SUBSCRIPTION_PLANS, create_customer, get_customer_by_user,
    create_subscription, get_active_subscription, create_payment_intent,
    record_usage, get_usage_stats, check_usage_limit, enforce_usage_limit,
    STRIPE_WEBHOOK_SECRET
)

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["Billing"], prefix="/billing")


# Customer Management Endpoints
@router.get("/customer", response_model=CustomerResponse)
async def get_customer_profile(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get current user's customer profile."""
    customer = get_customer_by_user(db, current_user)
    
    if not customer:
        # Auto-create customer if it doesn't exist
        customer_data = CustomerCreate(
            email=current_user.email,
            full_name=current_user.full_name
        )
        customer = create_customer(db, current_user, customer_data)
    
    return customer


@router.post("/customer", response_model=CustomerResponse, status_code=status.HTTP_201_CREATED)
async def create_customer_profile(
    customer_data: CustomerCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create customer profile for current user."""
    # Check if customer already exists
    existing_customer = get_customer_by_user(db, current_user)
    if existing_customer:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Customer profile already exists"
        )
    
    customer = create_customer(db, current_user, customer_data)
    return customer


# Subscription Management Endpoints
@router.get("/plans", response_model=List[SubscriptionPlan])
async def list_subscription_plans():
    """List all available subscription plans."""
    return list(SUBSCRIPTION_PLANS.values())


@router.get("/plans/{tier}", response_model=SubscriptionPlan)
async def get_subscription_plan(tier: SubscriptionTier):
    """Get specific subscription plan details."""
    if tier not in SUBSCRIPTION_PLANS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Subscription plan '{tier}' not found"
        )
    
    return SUBSCRIPTION_PLANS[tier]


@router.get("/subscription", response_model=Optional[SubscriptionResponse])
async def get_current_subscription(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get current user's active subscription."""
    customer = get_customer_by_user(db, current_user)
    
    if not customer:
        return None
    
    subscription = get_active_subscription(db, customer)
    return subscription


@router.post("/subscription", response_model=SubscriptionResponse, status_code=status.HTTP_201_CREATED)
async def create_user_subscription(
    subscription_data: SubscriptionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create or upgrade user subscription."""
    # Ensure customer exists
    customer = get_customer_by_user(db, current_user)
    if not customer:
        customer_data = CustomerCreate(
            email=current_user.email,
            full_name=current_user.full_name
        )
        customer = create_customer(db, current_user, customer_data)
    
    # Check for existing active subscription
    existing_subscription = get_active_subscription(db, customer)
    if existing_subscription and existing_subscription.tier != SubscriptionTier.FREE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Active subscription already exists. Please cancel current subscription first."
        )
    
    subscription = create_subscription(db, customer, subscription_data)
    return subscription


@router.put("/subscription/cancel")
async def cancel_subscription(
    cancel_at_period_end: bool = True,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Cancel current subscription."""
    customer = get_customer_by_user(db, current_user)
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found"
        )
    
    subscription = get_active_subscription(db, customer)
    if not subscription:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active subscription found"
        )
    
    if subscription.tier == SubscriptionTier.FREE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot cancel free subscription"
        )
    
    try:
        # Cancel in Stripe
        stripe.Subscription.modify(
            subscription.stripe_subscription_id,
            cancel_at_period_end=cancel_at_period_end
        )
        
        # Update database
        subscription.cancel_at_period_end = cancel_at_period_end
        if not cancel_at_period_end:
            subscription.status = SubscriptionStatus.CANCELLED
            subscription.cancelled_at = datetime.utcnow()
        
        db.commit()
        
        return {
            "message": "Subscription cancelled successfully",
            "cancel_at_period_end": cancel_at_period_end,
            "effective_date": subscription.current_period_end if cancel_at_period_end else datetime.utcnow()
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error cancelling subscription: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Stripe error: {str(e)}"
        )


# Payment Processing Endpoints
@router.post("/payment-intent")
async def create_payment(
    payment_data: PaymentIntentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create payment intent for one-time payments."""
    customer = get_customer_by_user(db, current_user)
    if not customer:
        customer_data = CustomerCreate(
            email=current_user.email,
            full_name=current_user.full_name
        )
        customer = create_customer(db, current_user, customer_data)
    
    payment_intent = create_payment_intent(db, customer, payment_data)
    return payment_intent


@router.get("/payments", response_model=List[PaymentResponse])
async def list_payments(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List user's payment history."""
    customer = get_customer_by_user(db, current_user)
    if not customer:
        return []
    
    payments = (db.query(Payment)
                .filter(Payment.customer_id == customer.id)
                .order_by(Payment.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all())
    
    return payments


# Usage Tracking Endpoints
@router.get("/usage", response_model=UsageStats)
async def get_usage_statistics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get current billing period usage statistics."""
    customer = get_customer_by_user(db, current_user)
    if not customer:
        # Return default stats for users without customer record
        from datetime import datetime
        now = datetime.utcnow()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        return UsageStats(
            billing_period_start=period_start,
            billing_period_end=period_start.replace(month=period_start.month + 1) if period_start.month < 12 else period_start.replace(year=period_start.year + 1, month=1),
            usage_by_type={usage_type: 0 for usage_type in UsageType},
            total_usage=0,
            limits={usage_type: SUBSCRIPTION_PLANS[SubscriptionTier.FREE].limits.get(usage_type.value, 0) for usage_type in UsageType},
            usage_remaining={usage_type: SUBSCRIPTION_PLANS[SubscriptionTier.FREE].limits.get(usage_type.value, 0) for usage_type in UsageType}
        )
    
    return get_usage_stats(db, customer)


@router.post("/usage/record", status_code=status.HTTP_201_CREATED)
async def record_usage_event(
    usage_data: UsageRecordCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Record usage event (typically called by system, not end users)."""
    customer = get_customer_by_user(db, current_user)
    if not customer:
        customer_data = CustomerCreate(
            email=current_user.email,
            full_name=current_user.full_name
        )
        customer = create_customer(db, current_user, customer_data)
    
    # Check usage limits before recording
    enforce_usage_limit(db, customer, usage_data.usage_type, usage_data.quantity)
    
    usage_record = record_usage(db, customer, usage_data)
    
    return {
        "message": "Usage recorded successfully",
        "usage_record_id": usage_record.id,
        "usage_type": usage_record.usage_type,
        "quantity": usage_record.quantity
    }


@router.get("/usage/check/{usage_type}")
async def check_usage_allowance(
    usage_type: UsageType,
    quantity: int = 1,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Check if user can perform specific usage action."""
    customer = get_customer_by_user(db, current_user)
    if not customer:
        # Default to free plan limits for users without customer record
        plan = SUBSCRIPTION_PLANS[SubscriptionTier.FREE]
        limit = plan.limits.get(usage_type.value, 0)
        allowed = limit == -1 or quantity <= limit  # Assume no prior usage for new users
    else:
        allowed = check_usage_limit(db, customer, usage_type, quantity)
    
    return {
        "allowed": allowed,
        "usage_type": usage_type,
        "requested_quantity": quantity
    }


# Webhook Endpoints
@router.post("/webhooks/stripe")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None, alias="stripe-signature"),
    db: Session = Depends(get_db)
):
    """Handle Stripe webhook events."""
    if not STRIPE_WEBHOOK_SECRET:
        logger.warning("Stripe webhook secret not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook secret not configured"
        )
    
    payload = await request.body()
    
    try:
        event = stripe.Webhook.construct_event(
            payload, stripe_signature, STRIPE_WEBHOOK_SECRET
        )
    except ValueError as e:
        logger.error(f"Invalid payload in webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid payload"
        )
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid signature in webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid signature"
        )
    
    # Handle the event
    try:
        if event["type"] == "customer.subscription.updated":
            await handle_subscription_updated(db, event["data"]["object"])
        elif event["type"] == "customer.subscription.deleted":
            await handle_subscription_deleted(db, event["data"]["object"])
        elif event["type"] == "invoice.payment_succeeded":
            await handle_payment_succeeded(db, event["data"]["object"])
        elif event["type"] == "invoice.payment_failed":
            await handle_payment_failed(db, event["data"]["object"])
        else:
            logger.info(f"Unhandled webhook event type: {event['type']}")
    
    except Exception as e:
        logger.error(f"Error handling webhook {event['type']}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing webhook"
        )
    
    return {"status": "success"}


# Admin Endpoints
@router.get("/admin/customers", response_model=List[CustomerResponse])
async def list_all_customers(
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """List all customers (admin only)."""
    customers = (db.query(Customer)
                .order_by(Customer.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all())
    
    return customers


@router.get("/admin/subscriptions", response_model=List[SubscriptionResponse])
async def list_all_subscriptions(
    tier: Optional[SubscriptionTier] = None,
    status: Optional[SubscriptionStatus] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """List all subscriptions with optional filtering (admin only)."""
    query = db.query(Subscription)
    
    if tier:
        query = query.filter(Subscription.tier == tier)
    if status:
        query = query.filter(Subscription.status == status)
    
    subscriptions = (query.order_by(Subscription.created_at.desc())
                    .offset(offset)
                    .limit(limit)
                    .all())
    
    return subscriptions


@router.get("/admin/revenue")
async def get_revenue_stats(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_admin_user)
):
    """Get revenue statistics (admin only)."""
    query = db.query(Payment).filter(Payment.status == PaymentStatus.SUCCESS)
    
    if start_date:
        query = query.filter(Payment.created_at >= start_date)
    if end_date:
        query = query.filter(Payment.created_at <= end_date)
    
    payments = query.all()
    
    total_revenue = sum(payment.amount for payment in payments)
    payment_count = len(payments)
    
    # Group by month for trends
    monthly_revenue = {}
    for payment in payments:
        month_key = payment.created_at.strftime("%Y-%m")
        monthly_revenue[month_key] = monthly_revenue.get(month_key, 0) + payment.amount
    
    return {
        "total_revenue": total_revenue,
        "payment_count": payment_count,
        "average_payment": total_revenue / payment_count if payment_count > 0 else 0,
        "monthly_breakdown": monthly_revenue
    }


# Webhook Event Handlers
async def handle_subscription_updated(db: Session, stripe_subscription: Dict[str, Any]):
    """Handle subscription updated webhook."""
    subscription = (db.query(Subscription)
                   .filter(Subscription.stripe_subscription_id == stripe_subscription["id"])
                   .first())
    
    if subscription:
        subscription.status = SubscriptionStatus(stripe_subscription["status"])
        subscription.current_period_start = datetime.fromtimestamp(stripe_subscription["current_period_start"])
        subscription.current_period_end = datetime.fromtimestamp(stripe_subscription["current_period_end"])
        subscription.cancel_at_period_end = stripe_subscription["cancel_at_period_end"]
        
        if stripe_subscription.get("canceled_at"):
            subscription.cancelled_at = datetime.fromtimestamp(stripe_subscription["canceled_at"])
        
        db.commit()
        logger.info(f"Updated subscription {subscription.id}")


async def handle_subscription_deleted(db: Session, stripe_subscription: Dict[str, Any]):
    """Handle subscription deleted webhook."""
    subscription = (db.query(Subscription)
                   .filter(Subscription.stripe_subscription_id == stripe_subscription["id"])
                   .first())
    
    if subscription:
        subscription.status = SubscriptionStatus.CANCELLED
        subscription.cancelled_at = datetime.utcnow()
        db.commit()
        logger.info(f"Cancelled subscription {subscription.id}")


async def handle_payment_succeeded(db: Session, stripe_invoice: Dict[str, Any]):
    """Handle successful payment webhook."""
    payment = (db.query(Payment)
              .filter(Payment.stripe_invoice_id == stripe_invoice["id"])
              .first())
    
    if payment:
        payment.status = PaymentStatus.SUCCESS
        db.commit()
        logger.info(f"Payment {payment.id} succeeded")


async def handle_payment_failed(db: Session, stripe_invoice: Dict[str, Any]):
    """Handle failed payment webhook."""
    payment = (db.query(Payment)
              .filter(Payment.stripe_invoice_id == stripe_invoice["id"])
              .first())
    
    if payment:
        payment.status = PaymentStatus.FAILED
        db.commit()
        logger.info(f"Payment {payment.id} failed") 
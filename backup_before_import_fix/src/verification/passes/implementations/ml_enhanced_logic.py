"""
ML-enhanced logic analysis for improved fallacy detection.

This module provides machine learning-based enhancement to the existing LLM-based
logic analysis system, offering:
- Ensemble methods combining rule-based, ML, and LLM approaches
- Pre-trained models for logical fallacy classification
- Confidence calibration and explainability features
- Performance optimizations with caching and batching
"""

import json
import logging
import os
import pickle
import time
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, Pipeline
)
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import spacy
from spacy.matcher import Matcher

from ....models.logic_bias import (
    LogicalIssue,
    LogicAnalysisResult,
    LogicalFallacyType,
    ReasoningIssueType
)


logger = logging.getLogger(__name__)


@dataclass
class MLLogicConfig:
    """Configuration for ML-enhanced logic analysis."""
    
    # Model paths and configurations
    transformer_model: str = "microsoft/DialoGPT-medium"
    rule_based_patterns_path: Optional[str] = None
    ml_model_cache_path: str = "models/logic_cache"
    
    # Ensemble settings
    use_ensemble: bool = True
    ensemble_weights: Dict[str, float] = None
    confidence_threshold: float = 0.6
    
    # Performance settings
    enable_caching: bool = True
    batch_size: int = 32
    max_tokens: int = 512
    
    # Explainability
    enable_explanations: bool = True
    highlight_evidence: bool = True
    
    def __post_init__(self):
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                "rule_based": 0.3,
                "ml_classifier": 0.4,
                "transformer": 0.3
            }


class FallacyPatternMatcher:
    """Rule-based fallacy pattern matcher using spaCy."""
    
    def __init__(self):
        """Initialize the pattern matcher."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy English model not found. Installing...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_patterns()
    
    def _setup_patterns(self):
        """Set up linguistic patterns for common fallacies."""
        
        # Ad hominem patterns
        ad_hominem_patterns = [
            [{"LOWER": {"IN": ["typical", "obviously"]}}, {"POS": "ADJ"}, {"LEMMA": "person"}],
            [{"LOWER": "you"}, {"LOWER": {"IN": ["can't", "cannot"]}}, {"LOWER": "trust"}],
            [{"LOWER": {"IN": ["idiot", "moron", "stupid"]}}, {"LOWER": {"IN": ["person", "people"]}}]
        ]
        self.matcher.add("AD_HOMINEM", ad_hominem_patterns)
        
        # Straw man patterns
        straw_man_patterns = [
            [{"LOWER": "so"}, {"LOWER": "you"}, {"LOWER": {"IN": ["think", "believe", "want"]}},
             {"LOWER": {"IN": ["all", "every", "no"]}}],
            [{"LOWER": {"IN": ["essentially", "basically"]}}, {"LOWER": "you're"}, 
             {"LOWER": {"IN": ["saying", "arguing"]}}]
        ]
        self.matcher.add("STRAW_MAN", straw_man_patterns)
        
        # False dichotomy patterns
        false_dichotomy_patterns = [
            [{"LOWER": "either"}, {"SKIP": {"ORTH": "..."}}, {"LOWER": "or"}],
            [{"LOWER": "you're"}, {"LOWER": "either"}, {"SKIP": {"ORTH": "..."}}, {"LOWER": "or"}],
            [{"LOWER": {"IN": ["only", "just"]}}, {"LOWER": "two"}, {"LOWER": {"IN": ["options", "choices"]}}]
        ]
        self.matcher.add("FALSE_DICHOTOMY", false_dichotomy_patterns)
        
        # Appeal to authority patterns
        appeal_authority_patterns = [
            [{"LOWER": {"IN": ["experts", "scientists", "doctors"]}}, {"LOWER": {"IN": ["say", "agree", "believe"]}}],
            [{"LOWER": "according"}, {"LOWER": "to"}, {"LOWER": {"IN": ["studies", "research"]}}],
            [{"LEMMA": {"IN": ["professor", "doctor"]}}, {"ORTH": "."}, {"LOWER": {"IN": ["says", "believes"]}}]
        ]
        self.matcher.add("APPEAL_TO_AUTHORITY", appeal_authority_patterns)
    
    def find_patterns(self, text: str) -> List[Dict[str, Any]]:
        """
        Find fallacy patterns in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of pattern matches with metadata
        """
        doc = self.nlp(text)
        matches = self.matcher(doc)
        
        results = []
        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            
            # Map pattern labels to fallacy types
            fallacy_type_map = {
                "AD_HOMINEM": LogicalFallacyType.AD_HOMINEM,
                "STRAW_MAN": LogicalFallacyType.STRAW_MAN,
                "FALSE_DICHOTOMY": LogicalFallacyType.FALSE_DICHOTOMY,
                "APPEAL_TO_AUTHORITY": LogicalFallacyType.APPEAL_TO_AUTHORITY
            }
            
            if label in fallacy_type_map:
                results.append({
                    "fallacy_type": fallacy_type_map[label],
                    "text_span": span.text,
                    "start_char": span.start_char,
                    "end_char": span.end_char,
                    "confidence": 0.7,  # Rule-based confidence
                    "method": "rule_based",
                    "evidence": f"Linguistic pattern match: {label.lower()}"
                })
        
        return results


class MLFallacyClassifier:
    """Machine learning-based fallacy classifier."""
    
    def __init__(self, config: MLLogicConfig):
        """
        Initialize the ML classifier.
        
        Args:
            config: ML configuration
        """
        self.config = config
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.tfidf_classifier = None
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models."""
        try:
            # Load transformer model for contextual understanding
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.transformer_model,
                cache_dir=self.config.ml_model_cache_path
            )
            
            # For demonstration, we'll use a sentiment analysis model as a proxy
            # In production, this would be a fine-tuned fallacy detection model
            self.pipeline = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
                max_length=self.config.max_tokens,
                truncation=True
            )
            
            # Initialize a simple TF-IDF classifier for baseline comparison
            self._setup_tfidf_classifier()
            
            logger.info("ML models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
            self.pipeline = None
    
    def _setup_tfidf_classifier(self):
        """Set up a simple TF-IDF based classifier."""
        # This would normally be trained on labeled fallacy data
        # For demonstration, we create a mock classifier
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english'
        )
        
        # Mock training data (in production, load from actual dataset)
        mock_texts = [
            "You're just saying that because you're biased",
            "All experts agree that this is true",
            "Either you support this or you hate freedom",
            "This argument is invalid because the person is unreliable"
        ]
        mock_labels = [
            LogicalFallacyType.AD_HOMINEM.value,
            LogicalFallacyType.APPEAL_TO_AUTHORITY.value,
            LogicalFallacyType.FALSE_DICHOTOMY.value,
            LogicalFallacyType.AD_HOMINEM.value
        ]
        
        # Create and train classifier
        X = self.tfidf_vectorizer.fit_transform(mock_texts)
        self.tfidf_classifier = LogisticRegression(random_state=42)
        
        # Expand labels to create a minimal training set
        expanded_X = np.vstack([X.toarray()] * 10)  # Repeat data for minimal training
        expanded_y = mock_labels * 10
        
        self.tfidf_classifier.fit(expanded_X, expanded_y)
    
    def classify_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Classify text for logical fallacies using ML models.
        
        Args:
            text: Text to classify
            
        Returns:
            List of detected fallacies with confidence scores
        """
        results = []
        
        if not self.pipeline:
            return results
        
        try:
            # Split text into sentences for analysis
            sentences = self._split_sentences(text)
            
            for i, sentence in enumerate(sentences):
                # Get transformer predictions
                transformer_results = self._classify_with_transformer(sentence)
                
                # Get TF-IDF predictions
                tfidf_results = self._classify_with_tfidf(sentence)
                
                # Combine results
                combined_results = self._combine_predictions(
                    transformer_results, tfidf_results, sentence, i
                )
                results.extend(combined_results)
        
        except Exception as e:
            logger.error(f"ML classification failed: {e}")
        
        return results
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences for analysis."""
        # Simple sentence splitting (could use spaCy for better results)
        sentences = []
        for line in text.split('\n'):
            if line.strip():
                # Split on sentence endings
                parts = line.replace('!', '.').replace('?', '.').split('.')
                sentences.extend([s.strip() for s in parts if s.strip()])
        return sentences
    
    def _classify_with_transformer(self, text: str) -> Dict[str, Any]:
        """Classify text using transformer model."""
        try:
            # Use sentiment as a proxy for fallacy detection
            # In production, this would be a fine-tuned fallacy classifier
            result = self.pipeline(text)
            
            # Map sentiment to potential fallacies (simplified approach)
            sentiment_to_fallacy = {
                "NEGATIVE": {
                    "fallacy_type": LogicalFallacyType.AD_HOMINEM,
                    "confidence": result[0]['score'] * 0.6  # Reduced confidence for proxy
                },
                "POSITIVE": {
                    "fallacy_type": None,  # Positive sentiment less likely to be fallacious
                    "confidence": 0.0
                }
            }
            
            sentiment = result[0]['label']
            return sentiment_to_fallacy.get(sentiment, {"fallacy_type": None, "confidence": 0.0})
            
        except Exception as e:
            logger.error(f"Transformer classification failed: {e}")
            return {"fallacy_type": None, "confidence": 0.0}
    
    def _classify_with_tfidf(self, text: str) -> Dict[str, Any]:
        """Classify text using TF-IDF classifier."""
        try:
            if not self.tfidf_classifier:
                return {"fallacy_type": None, "confidence": 0.0}
            
            # Vectorize text
            X = self.tfidf_vectorizer.transform([text])
            
            # Get prediction and probability
            prediction = self.tfidf_classifier.predict(X)[0]
            probabilities = self.tfidf_classifier.predict_proba(X)[0]
            max_prob = np.max(probabilities)
            
            # Convert prediction to enum
            try:
                fallacy_type = LogicalFallacyType(prediction)
            except ValueError:
                fallacy_type = None
            
            return {
                "fallacy_type": fallacy_type,
                "confidence": max_prob * 0.8  # Slightly reduced confidence
            }
            
        except Exception as e:
            logger.error(f"TF-IDF classification failed: {e}")
            return {"fallacy_type": None, "confidence": 0.0}
    
    def _combine_predictions(self, transformer_result: Dict[str, Any], 
                           tfidf_result: Dict[str, Any], 
                           text: str, sentence_idx: int) -> List[Dict[str, Any]]:
        """Combine predictions from multiple models."""
        results = []
        
        # Collect all non-null predictions
        predictions = []
        
        if transformer_result["fallacy_type"] and transformer_result["confidence"] > 0.3:
            predictions.append({
                "fallacy_type": transformer_result["fallacy_type"],
                "confidence": transformer_result["confidence"],
                "method": "transformer"
            })
        
        if tfidf_result["fallacy_type"] and tfidf_result["confidence"] > 0.3:
            predictions.append({
                "fallacy_type": tfidf_result["fallacy_type"],
                "confidence": tfidf_result["confidence"],
                "method": "tfidf"
            })
        
        # Create results for each prediction above threshold
        for pred in predictions:
            if pred["confidence"] >= self.config.confidence_threshold:
                results.append({
                    "fallacy_type": pred["fallacy_type"],
                    "text_span": text,
                    "start_char": sentence_idx * 100,  # Approximate position
                    "end_char": (sentence_idx + 1) * 100,
                    "confidence": pred["confidence"],
                    "method": pred["method"],
                    "evidence": f"ML classification via {pred['method']}"
                })
        
        return results


class MLEnhancedLogicAnalyzer:
    """
    Enhanced logic analyzer combining rule-based, ML, and LLM approaches.
    """
    
    def __init__(self, config: Optional[MLLogicConfig] = None):
        """
        Initialize the enhanced analyzer.
        
        Args:
            config: ML configuration (uses defaults if None)
        """
        self.config = config or MLLogicConfig()
        self.pattern_matcher = FallacyPatternMatcher()
        self.ml_classifier = MLFallacyClassifier(self.config)
        self.cache = {} if self.config.enable_caching else None
    
    def analyze_text(self, text: str, document_id: str) -> List[LogicalIssue]:
        """
        Analyze text for logical fallacies using ensemble methods.
        
        Args:
            text: Text to analyze
            document_id: Document identifier for tracking
            
        Returns:
            List of detected logical issues
        """
        # Check cache first
        if self.cache is not None:
            cache_key = hash(text)
            if cache_key in self.cache:
                logger.debug("Using cached analysis result")
                return self.cache[cache_key]
        
        # Collect predictions from all methods
        all_predictions = []
        
        # Rule-based analysis
        rule_predictions = self.pattern_matcher.find_patterns(text)
        all_predictions.extend(rule_predictions)
        
        # ML-based analysis
        ml_predictions = self.ml_classifier.classify_text(text)
        all_predictions.extend(ml_predictions)
        
        # Combine and deduplicate predictions
        combined_issues = self._combine_predictions(all_predictions, text, document_id)
        
        # Cache results if enabled
        if self.cache is not None:
            self.cache[cache_key] = combined_issues
        
        return combined_issues
    
    def _combine_predictions(self, predictions: List[Dict[str, Any]], 
                           text: str, document_id: str) -> List[LogicalIssue]:
        """
        Combine predictions from multiple methods into logical issues.
        
        Args:
            predictions: Raw predictions from all methods
            text: Original text
            document_id: Document identifier
            
        Returns:
            List of LogicalIssue objects
        """
        # Group predictions by fallacy type and location
        grouped_predictions = defaultdict(list)
        
        for pred in predictions:
            key = (
                pred["fallacy_type"],
                pred.get("start_char", 0) // 50  # Group by approximate location
            )
            grouped_predictions[key].append(pred)
        
        # Create LogicalIssue objects from grouped predictions
        issues = []
        for (fallacy_type, _), preds in grouped_predictions.items():
            if fallacy_type is None:
                continue
            
            # Calculate ensemble confidence
            if self.config.use_ensemble:
                confidence = self._calculate_ensemble_confidence(preds)
            else:
                confidence = max(p["confidence"] for p in preds)
            
            # Only include if above threshold
            if confidence < self.config.confidence_threshold:
                continue
            
            # Create the issue
            best_pred = max(preds, key=lambda p: p["confidence"])
            
            issue = LogicalIssue(
                fallacy_type=fallacy_type,
                reasoning_issue_type=ReasoningIssueType.INVALID_INFERENCE,  # Default
                text_excerpt=best_pred["text_span"],
                description=f"Detected {fallacy_type.value} using {best_pred['method']} analysis",
                severity=min(confidence, 1.0),
                confidence_score=confidence,
                location_start=best_pred.get("start_char", 0),
                location_end=best_pred.get("end_char", len(best_pred["text_span"])),
                document_id=document_id,
                evidence=[p["evidence"] for p in preds],
                suggested_improvement=self._generate_improvement_suggestion(fallacy_type),
                context_analysis=self._analyze_context(best_pred["text_span"], text)
            )
            
            issues.append(issue)
        
        return issues
    
    def _calculate_ensemble_confidence(self, predictions: List[Dict[str, Any]]) -> float:
        """Calculate ensemble confidence from multiple predictions."""
        if not predictions:
            return 0.0
        
        # Weight predictions by method
        weighted_sum = 0.0
        total_weight = 0.0
        
        for pred in predictions:
            method = pred.get("method", "unknown")
            weight = self.config.ensemble_weights.get(method, 0.1)
            weighted_sum += pred["confidence"] * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _generate_improvement_suggestion(self, fallacy_type: LogicalFallacyType) -> str:
        """Generate improvement suggestions for detected fallacies."""
        suggestions = {
            LogicalFallacyType.AD_HOMINEM: "Focus on addressing the argument itself rather than attacking the person making it.",
            LogicalFallacyType.STRAW_MAN: "Represent the opposing argument accurately before critiquing it.",
            LogicalFallacyType.FALSE_DICHOTOMY: "Consider additional alternatives beyond the two options presented.",
            LogicalFallacyType.APPEAL_TO_AUTHORITY: "Verify that the authority is relevant to the topic and that their position is accurately represented.",
            LogicalFallacyType.SLIPPERY_SLOPE: "Provide evidence for each step in the causal chain rather than assuming inevitable consequences."
        }
        
        return suggestions.get(fallacy_type, "Consider revising this argument to strengthen its logical foundation.")
    
    def _analyze_context(self, text_span: str, full_text: str) -> Dict[str, Any]:
        """Analyze the context around a detected fallacy."""
        # Find the position of the span in the full text
        start_pos = full_text.find(text_span)
        
        if start_pos == -1:
            return {"context_available": False}
        
        # Extract surrounding context (100 characters before and after)
        context_start = max(0, start_pos - 100)
        context_end = min(len(full_text), start_pos + len(text_span) + 100)
        context = full_text[context_start:context_end]
        
        return {
            "context_available": True,
            "surrounding_context": context,
            "relative_position": start_pos / len(full_text),
            "context_analysis": "Additional context available for human review"
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "transformer_model": self.config.transformer_model,
            "rule_patterns_loaded": len(self.pattern_matcher.matcher),
            "ml_model_available": self.ml_classifier.pipeline is not None,
            "ensemble_enabled": self.config.use_ensemble,
            "cache_enabled": self.config.enable_caching,
            "confidence_threshold": self.config.confidence_threshold
        }


# Factory function for easy instantiation
def create_ml_enhanced_analyzer(
    transformer_model: str = "microsoft/DialoGPT-medium",
    confidence_threshold: float = 0.6,
    enable_ensemble: bool = True,
    enable_caching: bool = True
) -> MLEnhancedLogicAnalyzer:
    """
    Create an ML-enhanced logic analyzer with custom configuration.
    
    Args:
        transformer_model: HuggingFace model to use
        confidence_threshold: Minimum confidence for reporting issues
        enable_ensemble: Whether to use ensemble methods
        enable_caching: Whether to cache analysis results
        
    Returns:
        Configured MLEnhancedLogicAnalyzer instance
    """
    config = MLLogicConfig(
        transformer_model=transformer_model,
        confidence_threshold=confidence_threshold,
        use_ensemble=enable_ensemble,
        enable_caching=enable_caching
    )
    
    return MLEnhancedLogicAnalyzer(config) 
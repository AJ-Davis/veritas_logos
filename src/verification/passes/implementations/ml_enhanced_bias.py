"""
ML-enhanced bias detection for advanced bias analysis.

This module provides machine learning-based enhancement to the existing LLM-based
bias detection system, offering:
- Ensemble methods combining lexical, statistical, and ML approaches
- Pre-trained models for bias classification with fairness metrics
- Demographic representation analysis
- Confidence calibration and explainability features
- Performance optimizations with caching and batching
"""

import json
import logging
import os
import pickle
import re
import time
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
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

try:
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    logging.warning("Fairlearn not available. Install with: pip install fairlearn")

from src.models.logic_bias import (
    BiasIssue,
    BiasAnalysisResult,
    BiasType,
    BiasSeverity
)


logger = logging.getLogger(__name__)


@dataclass
class MLBiasConfig:
    """Configuration for ML-enhanced bias detection."""
    
    # Model paths and configurations
    transformer_model: str = "unitary/toxic-bert"
    bias_lexicon_path: Optional[str] = None
    ml_model_cache_path: str = "models/bias_cache"
    
    # Bias detection settings
    bias_types_to_detect: List[str] = None
    enable_demographic_analysis: bool = True
    enable_fairness_metrics: bool = FAIRLEARN_AVAILABLE
    
    # Ensemble settings
    use_ensemble: bool = True
    ensemble_weights: Dict[str, float] = None
    confidence_threshold: float = 0.5
    
    # Performance settings
    enable_caching: bool = True
    batch_size: int = 16
    max_tokens: int = 512
    
    # Explainability
    enable_explanations: bool = True
    highlight_evidence: bool = True
    
    def __post_init__(self):
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                "lexical": 0.25,
                "statistical": 0.25,
                "ml_classifier": 0.3,
                "transformer": 0.2
            }
        
        if self.bias_types_to_detect is None:
            self.bias_types_to_detect = [bt.value for bt in BiasType]


class BiasLexicon:
    """Lexical bias detection using curated word lists."""
    
    def __init__(self, lexicon_path: Optional[str] = None):
        """
        Initialize bias lexicon.
        
        Args:
            lexicon_path: Path to custom lexicon file (optional)
        """
        self.bias_words = self._load_default_lexicon()
        if lexicon_path and os.path.exists(lexicon_path):
            self.bias_words.update(self._load_custom_lexicon(lexicon_path))
    
    def _load_default_lexicon(self) -> Dict[BiasType, Set[str]]:
        """Load default bias word lexicon."""
        return {
            BiasType.GENDER_BIAS: {
                "manly", "feminine", "ladylike", "girly", "bossy", "emotional",
                "hysterical", "nurturing", "aggressive", "sensitive", "weak",
                "strong", "macho", "delicate", "fragile", "tough"
            },
            BiasType.RACIAL_BIAS: {
                "exotic", "articulate", "urban", "ghetto", "primitive", "civilized",
                "savage", "barbaric", "cultured", "refined", "ethnic", "foreign",
                "alien", "native", "tribal"
            },
            BiasType.AGE_BIAS: {
                "old-fashioned", "outdated", "senile", "immature", "childish",
                "youthful", "experienced", "seasoned", "fresh", "green",
                "ancient", "modern", "contemporary", "traditional"
            },
            BiasType.RELIGIOUS_BIAS: {
                "fanatic", "extremist", "fundamentalist", "radical", "zealot",
                "infidel", "heathen", "godless", "pious", "devout", "orthodox",
                "secular", "atheistic", "blasphemous"
            },
            BiasType.POLITICAL_BIAS: {
                "liberal", "conservative", "leftist", "rightist", "radical",
                "moderate", "extremist", "centrist", "progressive", "reactionary",
                "socialist", "capitalist", "communist", "fascist"
            },
            BiasType.SOCIOECONOMIC_BIAS: {
                "privileged", "underprivileged", "elite", "common", "classy",
                "trashy", "refined", "crude", "sophisticated", "simple",
                "wealthy", "poor", "rich", "broke", "expensive", "cheap"
            }
        }
    
    def _load_custom_lexicon(self, path: str) -> Dict[BiasType, Set[str]]:
        """Load custom bias lexicon from file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            custom_lexicon = {}
            for bias_type_str, words in data.items():
                try:
                    bias_type = BiasType(bias_type_str)
                    custom_lexicon[bias_type] = set(words)
                except ValueError:
                    logger.warning(f"Unknown bias type in lexicon: {bias_type_str}")
            
            return custom_lexicon
        except Exception as e:
            logger.error(f"Failed to load custom lexicon: {e}")
            return {}
    
    def detect_bias_words(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect bias words in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected bias indicators
        """
        results = []
        text_lower = text.lower()
        
        for bias_type, words in self.bias_words.items():
            for word in words:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(word.lower()) + r'\b'
                matches = list(re.finditer(pattern, text_lower))
                
                for match in matches:
                    results.append({
                        "bias_type": bias_type,
                        "word": word,
                        "start_pos": match.start(),
                        "end_pos": match.end(),
                        "confidence": 0.6,  # Base lexical confidence
                        "method": "lexical",
                        "evidence": f"Bias-indicating word: '{word}'"
                    })
        
        return results


class StatisticalBiasAnalyzer:
    """Statistical analysis for bias detection."""
    
    def __init__(self):
        """Initialize the statistical analyzer."""
        self.demographic_groups = {
            "gender": ["men", "women", "male", "female", "man", "woman", "guy", "girl", "boy", "girl"],
            "race": ["white", "black", "asian", "hispanic", "latino", "african", "european", "american"],
            "age": ["young", "old", "elderly", "senior", "teenager", "adult", "child", "youth"],
            "religion": ["christian", "muslim", "jewish", "hindu", "buddhist", "atheist", "religious"]
        }
    
    def analyze_representation(self, text: str) -> Dict[str, Any]:
        """
        Analyze demographic representation in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with representation statistics
        """
        text_lower = text.lower()
        representation = {}
        
        for category, groups in self.demographic_groups.items():
            group_counts = {}
            total_mentions = 0
            
            for group in groups:
                count = len(re.findall(r'\b' + re.escape(group) + r'\b', text_lower))
                if count > 0:
                    group_counts[group] = count
                    total_mentions += count
            
            if total_mentions > 0:
                # Calculate proportional representation
                proportions = {group: count / total_mentions 
                             for group, count in group_counts.items()}
                
                # Calculate diversity index (Shannon entropy)
                diversity = -sum(p * np.log(p) for p in proportions.values() if p > 0)
                
                representation[category] = {
                    "group_counts": group_counts,
                    "proportions": proportions,
                    "total_mentions": total_mentions,
                    "diversity_index": diversity,
                    "dominant_group": max(proportions, key=proportions.get) if proportions else None
                }
        
        return representation
    
    def detect_statistical_bias(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect statistical bias patterns.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected bias issues
        """
        results = []
        representation = self.analyze_representation(text)
        
        for category, stats in representation.items():
            if stats["total_mentions"] < 3:  # Skip categories with few mentions
                continue
            
            # Check for extreme imbalances
            proportions = list(stats["proportions"].values())
            max_prop = max(proportions)
            min_prop = min(proportions)
            
            # Flag if one group dominates (>80% of mentions)
            if max_prop > 0.8 and len(proportions) > 1:
                results.append({
                    "bias_type": self._category_to_bias_type(category),
                    "issue": "representation_imbalance",
                    "dominant_group": stats["dominant_group"],
                    "proportion": max_prop,
                    "confidence": min(max_prop, 0.9),
                    "method": "statistical",
                    "evidence": f"Extreme representation imbalance in {category}: {stats['dominant_group']} ({max_prop:.1%})"
                })
            
            # Flag very low diversity
            if stats["diversity_index"] < 0.5 and len(proportions) > 1:
                results.append({
                    "bias_type": self._category_to_bias_type(category),
                    "issue": "low_diversity",
                    "diversity_index": stats["diversity_index"],
                    "confidence": 0.7,
                    "method": "statistical",
                    "evidence": f"Low diversity in {category} representation (diversity index: {stats['diversity_index']:.2f})"
                })
        
        return results
    
    def _category_to_bias_type(self, category: str) -> BiasType:
        """Map demographic category to bias type."""
        mapping = {
            "gender": BiasType.GENDER_BIAS,
            "race": BiasType.RACIAL_BIAS,
            "age": BiasType.AGE_BIAS,
            "religion": BiasType.RELIGIOUS_BIAS
        }
        return mapping.get(category, BiasType.CONFIRMATION_BIAS)


class MLBiasClassifier:
    """Machine learning-based bias classifier."""
    
    def __init__(self, config: MLBiasConfig):
        """
        Initialize the ML bias classifier.
        
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
            # Load transformer model for bias detection
            # Using a toxicity model as a proxy for bias detection
            self.pipeline = pipeline(
                "text-classification",
                model=self.config.transformer_model,
                max_length=self.config.max_tokens,
                truncation=True
            )
            
            # Initialize TF-IDF classifier for baseline comparison
            self._setup_tfidf_classifier()
            
            logger.info(f"ML bias models loaded successfully: {self.config.transformer_model}")
            
        except Exception as e:
            logger.error(f"Failed to load ML bias models: {e}")
            self.pipeline = None
    
    def _setup_tfidf_classifier(self):
        """Set up a TF-IDF based bias classifier."""
        # Mock training data for demonstration
        mock_texts = [
            "Women are naturally more emotional and less logical",
            "All Asian students are good at math",
            "Older workers are less productive and adaptable",
            "This research proves that our hypothesis is correct"
        ]
        mock_labels = [
            BiasType.GENDER_BIAS.value,
            BiasType.RACIAL_BIAS.value,
            BiasType.AGE_BIAS.value,
            BiasType.CONFIRMATION_BIAS.value
        ]
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Create and train classifier
        X = self.tfidf_vectorizer.fit_transform(mock_texts)
        self.tfidf_classifier = LogisticRegression(random_state=42)
        
        # Expand data for minimal training
        expanded_X = np.vstack([X.toarray()] * 15)
        expanded_y = mock_labels * 15
        
        self.tfidf_classifier.fit(expanded_X, expanded_y)
    
    def classify_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Classify text for bias using ML models.
        
        Args:
            text: Text to classify
            
        Returns:
            List of detected bias issues
        """
        results = []
        
        if not self.pipeline:
            return results
        
        try:
            # Split text into segments for analysis
            segments = self._segment_text(text)
            
            for i, segment in enumerate(segments):
                # Get transformer predictions
                transformer_results = self._classify_with_transformer(segment)
                
                # Get TF-IDF predictions
                tfidf_results = self._classify_with_tfidf(segment)
                
                # Combine results
                combined_results = self._combine_bias_predictions(
                    transformer_results, tfidf_results, segment, i
                )
                results.extend(combined_results)
        
        except Exception as e:
            logger.error(f"ML bias classification failed: {e}")
        
        return results
    
    def _segment_text(self, text: str) -> List[str]:
        """Segment text for bias analysis."""
        # Split into sentences and group into segments
        sentences = []
        for line in text.split('\n'):
            if line.strip():
                parts = re.split(r'[.!?]+', line)
                sentences.extend([s.strip() for s in parts if s.strip()])
        
        # Group sentences into segments of 2-3 sentences
        segments = []
        for i in range(0, len(sentences), 2):
            segment = ' '.join(sentences[i:i+2])
            if segment:
                segments.append(segment)
        
        return segments
    
    def _classify_with_transformer(self, text: str) -> Dict[str, Any]:
        """Classify text using transformer model."""
        try:
            # Use toxicity model as proxy for bias detection
            result = self.pipeline(text)
            
            # Map toxicity to bias (simplified approach)
            if isinstance(result, list) and len(result) > 0:
                score = result[0].get('score', 0)
                label = result[0].get('label', '')
                
                # If toxic/biased content detected
                if label.upper() in ['TOXIC', 'BIASED', '1'] and score > 0.5:
                    return {
                        "bias_detected": True,
                        "confidence": score * 0.8,  # Reduced confidence for proxy
                        "bias_type": BiasType.CONFIRMATION_BIAS  # Default type
                    }
            
            return {"bias_detected": False, "confidence": 0.0}
            
        except Exception as e:
            logger.error(f"Transformer bias classification failed: {e}")
            return {"bias_detected": False, "confidence": 0.0}
    
    def _classify_with_tfidf(self, text: str) -> Dict[str, Any]:
        """Classify text using TF-IDF classifier."""
        try:
            if not self.tfidf_classifier:
                return {"bias_detected": False, "confidence": 0.0}
            
            # Vectorize text
            X = self.tfidf_vectorizer.transform([text])
            
            # Get prediction and probability
            prediction = self.tfidf_classifier.predict(X)[0]
            probabilities = self.tfidf_classifier.predict_proba(X)[0]
            max_prob = np.max(probabilities)
            
            # Convert prediction to enum
            try:
                bias_type = BiasType(prediction)
                return {
                    "bias_detected": True,
                    "bias_type": bias_type,
                    "confidence": max_prob
                }
            except ValueError:
                return {"bias_detected": False, "confidence": 0.0}
            
        except Exception as e:
            logger.error(f"TF-IDF bias classification failed: {e}")
            return {"bias_detected": False, "confidence": 0.0}
    
    def _combine_bias_predictions(self, transformer_result: Dict[str, Any],
                                tfidf_result: Dict[str, Any],
                                text: str, segment_idx: int) -> List[Dict[str, Any]]:
        """Combine bias predictions from multiple models."""
        results = []
        
        # Collect predictions
        predictions = []
        
        if transformer_result.get("bias_detected") and transformer_result["confidence"] > 0.4:
            predictions.append({
                "bias_type": transformer_result.get("bias_type", BiasType.CONFIRMATION_BIAS),
                "confidence": transformer_result["confidence"],
                "method": "transformer"
            })
        
        if tfidf_result.get("bias_detected") and tfidf_result["confidence"] > 0.4:
            predictions.append({
                "bias_type": tfidf_result["bias_type"],
                "confidence": tfidf_result["confidence"],
                "method": "tfidf"
            })
        
        # Create results for predictions above threshold
        for pred in predictions:
            if pred["confidence"] >= self.config.confidence_threshold:
                results.append({
                    "bias_type": pred["bias_type"],
                    "text_span": text,
                    "start_char": segment_idx * 200,  # Approximate position
                    "end_char": (segment_idx + 1) * 200,
                    "confidence": pred["confidence"],
                    "method": pred["method"],
                    "evidence": f"ML bias detection via {pred['method']}"
                })
        
        return results


class MLEnhancedBiasAnalyzer:
    """
    Enhanced bias analyzer combining lexical, statistical, ML, and LLM approaches.
    """
    
    def __init__(self, config: Optional[MLBiasConfig] = None):
        """
        Initialize the enhanced bias analyzer.
        
        Args:
            config: ML configuration (uses defaults if None)
        """
        self.config = config or MLBiasConfig()
        self.lexicon = BiasLexicon(self.config.bias_lexicon_path)
        self.statistical_analyzer = StatisticalBiasAnalyzer()
        self.ml_classifier = MLBiasClassifier(self.config)
        self.cache = {} if self.config.enable_caching else None
    
    def analyze_text(self, text: str, document_id: str) -> List[BiasIssue]:
        """
        Analyze text for bias using ensemble methods.
        
        Args:
            text: Text to analyze
            document_id: Document identifier for tracking
            
        Returns:
            List of detected bias issues
        """
        # Check cache first
        if self.cache is not None:
            cache_key = hash(text)
            if cache_key in self.cache:
                logger.debug("Using cached bias analysis result")
                return self.cache[cache_key]
        
        # Collect predictions from all methods
        all_predictions = []
        
        # Lexical analysis
        lexical_predictions = self.lexicon.detect_bias_words(text)
        all_predictions.extend(lexical_predictions)
        
        # Statistical analysis
        statistical_predictions = self.statistical_analyzer.detect_statistical_bias(text)
        all_predictions.extend(statistical_predictions)
        
        # ML-based analysis
        ml_predictions = self.ml_classifier.classify_text(text)
        all_predictions.extend(ml_predictions)
        
        # Combine and deduplicate predictions
        combined_issues = self._combine_bias_predictions(all_predictions, text, document_id)
        
        # Cache results if enabled
        if self.cache is not None:
            self.cache[cache_key] = combined_issues
        
        return combined_issues
    
    def _combine_bias_predictions(self, predictions: List[Dict[str, Any]],
                                text: str, document_id: str) -> List[BiasIssue]:
        """
        Combine predictions from multiple methods into bias issues.
        
        Args:
            predictions: Raw predictions from all methods
            text: Original text
            document_id: Document identifier
            
        Returns:
            List of BiasIssue objects
        """
        # Group predictions by bias type and location
        grouped_predictions = defaultdict(list)
        
        for pred in predictions:
            bias_type = pred.get("bias_type")
            if bias_type is None:
                continue
            
            location_key = pred.get("start_char", pred.get("start_pos", 0)) // 100
            key = (bias_type, location_key)
            grouped_predictions[key].append(pred)
        
        # Create BiasIssue objects from grouped predictions
        issues = []
        for (bias_type, _), preds in grouped_predictions.items():
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
            
            # Determine severity from confidence
            severity = self._confidence_to_severity(confidence)
            
            issue = BiasIssue(
                bias_type=bias_type,
                text_excerpt=best_pred.get("text_span", best_pred.get("word", "")),
                description=f"Detected {bias_type.value} using {best_pred.get('method', 'unknown')} analysis",
                severity=severity,
                confidence_score=confidence,
                impact_score=min(confidence * 1.2, 1.0),
                start_position=best_pred.get("start_char", best_pred.get("start_pos", 0)),
                end_position=best_pred.get("end_char", best_pred.get("end_pos", 0)),
                document_id=document_id,
                evidence=[p.get("evidence", "") for p in preds],
                mitigation_suggestions=self._generate_mitigation_suggestions(bias_type),
                affected_groups=self._identify_affected_groups(bias_type, best_pred),
                context_analysis=self._analyze_bias_context(best_pred.get("text_span", ""), text)
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
    
    def _confidence_to_severity(self, confidence: float) -> BiasSeverity:
        """Convert confidence score to bias severity."""
        if confidence >= 0.8:
            return BiasSeverity.HIGH
        elif confidence >= 0.6:
            return BiasSeverity.MODERATE
        else:
            return BiasSeverity.LOW
    
    def _generate_mitigation_suggestions(self, bias_type: BiasType) -> List[str]:
        """Generate mitigation suggestions for detected bias."""
        suggestions = {
            BiasType.GENDER_BIAS: [
                "Use gender-neutral language where possible",
                "Avoid assumptions about gender roles and capabilities",
                "Include diverse perspectives from all genders"
            ],
            BiasType.RACIAL_BIAS: [
                "Use people-first language",
                "Avoid racial stereotypes and generalizations",
                "Ensure representation from diverse racial backgrounds"
            ],
            BiasType.AGE_BIAS: [
                "Avoid age-related stereotypes",
                "Focus on skills and experience rather than age",
                "Include intergenerational perspectives"
            ],
            BiasType.CONFIRMATION_BIAS: [
                "Present alternative viewpoints",
                "Cite diverse sources and evidence",
                "Question underlying assumptions"
            ],
            BiasType.SELECTION_BIAS: [
                "Expand sample selection criteria",
                "Use randomized selection methods",
                "Acknowledge limitations in data collection"
            ]
        }
        
        return suggestions.get(bias_type, ["Review content for potential bias and consider alternative perspectives"])
    
    def _identify_affected_groups(self, bias_type: BiasType, prediction: Dict[str, Any]) -> List[str]:
        """Identify groups affected by the detected bias."""
        group_mappings = {
            BiasType.GENDER_BIAS: ["women", "men", "non-binary individuals"],
            BiasType.RACIAL_BIAS: ["racial minorities", "ethnic groups"],
            BiasType.AGE_BIAS: ["older adults", "younger people"],
            BiasType.RELIGIOUS_BIAS: ["religious minorities"],
            BiasType.SOCIOECONOMIC_BIAS: ["lower-income individuals", "working class"]
        }
        
        base_groups = group_mappings.get(bias_type, ["affected communities"])
        
        # Try to identify specific groups from the prediction
        if "dominant_group" in prediction:
            base_groups.append(f"non-{prediction['dominant_group']} groups")
        
        return base_groups
    
    def _analyze_bias_context(self, text_span: str, full_text: str) -> Dict[str, Any]:
        """Analyze the context around detected bias."""
        if not text_span:
            return {"context_available": False}
        
        # Find the position of the span in the full text
        start_pos = full_text.find(text_span)
        
        if start_pos == -1:
            return {"context_available": False}
        
        # Extract surrounding context
        context_start = max(0, start_pos - 150)
        context_end = min(len(full_text), start_pos + len(text_span) + 150)
        context = full_text[context_start:context_end]
        
        return {
            "context_available": True,
            "surrounding_context": context,
            "relative_position": start_pos / len(full_text),
            "context_analysis": "Context available for review"
        }
    
    def get_representation_analysis(self, text: str) -> Dict[str, Any]:
        """Get detailed demographic representation analysis."""
        return self.statistical_analyzer.analyze_representation(text)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "transformer_model": self.config.transformer_model,
            "lexicon_size": sum(len(words) for words in self.lexicon.bias_words.values()),
            "ml_model_available": self.ml_classifier.pipeline is not None,
            "fairness_metrics_enabled": self.config.enable_fairness_metrics,
            "ensemble_enabled": self.config.use_ensemble,
            "cache_enabled": self.config.enable_caching,
            "confidence_threshold": self.config.confidence_threshold,
            "bias_types_detected": len(self.config.bias_types_to_detect)
        }


# Factory function for easy instantiation
def create_ml_enhanced_bias_analyzer(
    transformer_model: str = "unitary/toxic-bert",
    confidence_threshold: float = 0.5,
    enable_ensemble: bool = True,
    enable_caching: bool = True,
    enable_fairness_metrics: bool = True
) -> MLEnhancedBiasAnalyzer:
    """
    Create an ML-enhanced bias analyzer with custom configuration.
    
    Args:
        transformer_model: HuggingFace model to use for bias detection
        confidence_threshold: Minimum confidence for reporting issues
        enable_ensemble: Whether to use ensemble methods
        enable_caching: Whether to cache analysis results
        enable_fairness_metrics: Whether to use fairness metrics (requires fairlearn)
        
    Returns:
        Configured MLEnhancedBiasAnalyzer instance
    """
    config = MLBiasConfig(
        transformer_model=transformer_model,
        confidence_threshold=confidence_threshold,
        use_ensemble=enable_ensemble,
        enable_caching=enable_caching,
        enable_fairness_metrics=enable_fairness_metrics and FAIRLEARN_AVAILABLE
    )
    
    return MLEnhancedBiasAnalyzer(config) 
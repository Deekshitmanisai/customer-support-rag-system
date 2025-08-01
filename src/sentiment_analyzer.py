import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os

class SentimentAnalyzer:
    """Comprehensive sentiment analysis with emotion detection and escalation prediction."""
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.emotion_model = None
        self.emotion_vectorizer = None
        self.emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'trust', 'anticipation']
        self._load_or_train_emotion_model()
        
        # Escalation keywords and patterns
        self.escalation_keywords = {
            'high': ['furious', 'outraged', 'unacceptable', 'terrible', 'horrible', 'awful', 'disgusting'],
            'medium': ['frustrated', 'annoyed', 'disappointed', 'upset', 'angry', 'mad'],
            'low': ['slightly', 'a bit', 'kind of', 'somewhat']
        }
        
        self.urgency_indicators = [
            'urgent', 'immediately', 'asap', 'right now', 'emergency', 'critical',
            'cannot wait', 'need help now', 'desperate', 'urgently'
        ]
    
    def _load_or_train_emotion_model(self):
        """Load pre-trained emotion model or train a new one."""
        model_path = "models/emotion_model.pkl"
        vectorizer_path = "models/emotion_vectorizer.pkl"
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            try:
                with open(model_path, 'rb') as f:
                    self.emotion_model = pickle.load(f)
                with open(vectorizer_path, 'rb') as f:
                    self.emotion_vectorizer = pickle.load(f)
                return
            except:
                pass
        
        # Train a simple emotion model with sample data
        self._train_emotion_model()
    
    def _train_emotion_model(self):
        """Train emotion classification model with sample data."""
        # Sample training data for emotions
        training_data = {
            'joy': [
                "I'm so happy with the service!", "Thank you so much!", "This is wonderful!",
                "Great job!", "Excellent service!", "I love this!", "Amazing support!"
            ],
            'sadness': [
                "I'm really disappointed", "This makes me sad", "I'm upset about this",
                "This is heartbreaking", "I feel let down", "This is depressing"
            ],
            'anger': [
                "I'm furious about this!", "This is unacceptable!", "I'm so angry!",
                "This makes me mad!", "I'm outraged!", "This is terrible!"
            ],
            'fear': [
                "I'm worried about this", "This is scary", "I'm afraid of what might happen",
                "This concerns me", "I'm anxious about this", "This is frightening"
            ],
            'surprise': [
                "Wow, I didn't expect this!", "This is surprising!", "I'm shocked!",
                "This came out of nowhere!", "Unexpected!", "I'm amazed!"
            ],
            'disgust': [
                "This is disgusting", "I'm appalled", "This is revolting",
                "This is sickening", "I'm repulsed by this"
            ],
            'trust': [
                "I trust you to help", "I believe in your service", "I have confidence in you",
                "I rely on you", "I trust your expertise"
            ],
            'anticipation': [
                "I'm looking forward to this", "I can't wait", "I'm excited about this",
                "I'm anticipating good results", "I'm hopeful"
            ]
        }
        
        # Prepare training data
        texts = []
        labels = []
        
        for emotion, samples in training_data.items():
            texts.extend(samples)
            labels.extend([emotion] * len(samples))
        
        # Vectorize text
        self.emotion_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = self.emotion_vectorizer.fit_transform(texts)
        
        # Train model
        self.emotion_model = MultinomialNB()
        self.emotion_model.fit(X, labels)
        
        # Save models
        os.makedirs("models", exist_ok=True)
        with open("models/emotion_model.pkl", 'wb') as f:
            pickle.dump(self.emotion_model, f)
        with open("models/emotion_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.emotion_vectorizer, f)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Comprehensive sentiment analysis with multiple models."""
        text = text.strip()
        if not text:
            return self._empty_sentiment_result()
        
        # VADER sentiment analysis
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        # TextBlob sentiment analysis
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Emotion detection
        emotions = self.detect_emotions(text)
        
        # Escalation analysis
        escalation_score = self.analyze_escalation_risk(text)
        
        # Overall sentiment classification
        overall_sentiment = self._classify_overall_sentiment(vader_scores, textblob_polarity)
        
        return {
            'text': text,
            'overall_sentiment': overall_sentiment,
            'vader_scores': vader_scores,
            'textblob_scores': {
                'polarity': textblob_polarity,
                'subjectivity': textblob_subjectivity
            },
            'emotions': emotions,
            'escalation_risk': escalation_score,
            'urgency_level': self._detect_urgency(text),
            'confidence': self._calculate_confidence(vader_scores, textblob_polarity)
        }
    
    def detect_emotions(self, text: str) -> Dict[str, float]:
        """Detect emotions in text using trained model."""
        if not self.emotion_model or not self.emotion_vectorizer:
            return {emotion: 0.0 for emotion in self.emotions}
        
        try:
            # Vectorize input text
            X = self.emotion_vectorizer.transform([text])
            
            # Get probability predictions
            probabilities = self.emotion_model.predict_proba(X)[0]
            
            # Create emotion dictionary
            emotion_scores = {}
            for i, emotion in enumerate(self.emotions):
                if i < len(probabilities):
                    emotion_scores[emotion] = float(probabilities[i])
                else:
                    emotion_scores[emotion] = 0.0
            
            return emotion_scores
        except:
            return {emotion: 0.0 for emotion in self.emotions}
    
    def analyze_escalation_risk(self, text: str) -> Dict[str, Any]:
        """Analyze escalation risk based on keywords and patterns."""
        text_lower = text.lower()
        
        # Count escalation keywords
        risk_scores = {}
        for level, keywords in self.escalation_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            risk_scores[level] = count
        
        # Calculate overall risk score
        high_risk = risk_scores['high'] * 3
        medium_risk = risk_scores['medium'] * 2
        low_risk = risk_scores['low'] * 1
        
        total_risk = high_risk + medium_risk + low_risk
        
        # Normalize to 0-1 scale
        max_possible_risk = 10  # Arbitrary max
        normalized_risk = min(total_risk / max_possible_risk, 1.0)
        
        # Determine risk level
        if normalized_risk >= 0.7:
            risk_level = "high"
        elif normalized_risk >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            'risk_score': normalized_risk,
            'risk_level': risk_level,
            'keyword_counts': risk_scores,
            'total_risk': total_risk
        }
    
    def _detect_urgency(self, text: str) -> str:
        """Detect urgency level in text."""
        text_lower = text.lower()
        urgency_count = sum(1 for indicator in self.urgency_indicators if indicator in text_lower)
        
        if urgency_count >= 2:
            return "high"
        elif urgency_count == 1:
            return "medium"
        else:
            return "low"
    
    def _classify_overall_sentiment(self, vader_scores: Dict, textblob_polarity: float) -> str:
        """Classify overall sentiment based on multiple models."""
        vader_compound = vader_scores['compound']
        
        # Combine VADER and TextBlob scores
        combined_score = (vader_compound + textblob_polarity) / 2
        
        if combined_score >= 0.1:
            return "positive"
        elif combined_score <= -0.1:
            return "negative"
        else:
            return "neutral"
    
    def _calculate_confidence(self, vader_scores: Dict, textblob_polarity: float) -> float:
        """Calculate confidence in sentiment analysis."""
        # Use VADER's compound score magnitude as confidence
        confidence = abs(vader_scores['compound'])
        return min(confidence, 1.0)
    
    def _empty_sentiment_result(self) -> Dict[str, Any]:
        """Return empty sentiment result for empty text."""
        return {
            'text': '',
            'overall_sentiment': 'neutral',
            'vader_scores': {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0},
            'textblob_scores': {'polarity': 0.0, 'subjectivity': 0.0},
            'emotions': {emotion: 0.0 for emotion in self.emotions},
            'escalation_risk': {
                'risk_score': 0.0,
                'risk_level': 'low',
                'keyword_counts': {'high': 0, 'medium': 0, 'low': 0},
                'total_risk': 0
            },
            'urgency_level': 'low',
            'confidence': 0.0
        }
    
    def analyze_conversation_sentiment(self, messages: List[Dict]) -> Dict[str, Any]:
        """Analyze sentiment progression in a conversation."""
        if not messages:
            return {'trend': 'stable', 'overall_sentiment': 'neutral', 'escalation_trend': 'stable'}
        
        sentiments = []
        escalation_scores = []
        
        for message in messages:
            if message.get('role') == 'user':
                analysis = self.analyze_sentiment(message.get('content', ''))
                sentiments.append(analysis['overall_sentiment'])
                escalation_scores.append(analysis['escalation_risk']['risk_score'])
        
        # Analyze trends
        sentiment_trend = self._analyze_sentiment_trend(sentiments)
        escalation_trend = self._analyze_escalation_trend(escalation_scores)
        
        return {
            'trend': sentiment_trend,
            'overall_sentiment': self._get_most_common(sentiments),
            'escalation_trend': escalation_trend,
            'message_count': len(messages),
            'sentiment_history': sentiments,
            'escalation_history': escalation_scores
        }
    
    def _analyze_sentiment_trend(self, sentiments: List[str]) -> str:
        """Analyze sentiment trend in conversation."""
        if len(sentiments) < 2:
            return 'stable'
        
        # Simple trend analysis
        positive_count = sentiments.count('positive')
        negative_count = sentiments.count('negative')
        
        if positive_count > negative_count:
            return 'improving'
        elif negative_count > positive_count:
            return 'deteriorating'
        else:
            return 'stable'
    
    def _analyze_escalation_trend(self, escalation_scores: List[float]) -> str:
        """Analyze escalation trend in conversation."""
        if len(escalation_scores) < 2:
            return 'stable'
        
        # Check if escalation scores are increasing
        if len(escalation_scores) >= 3:
            recent_avg = sum(escalation_scores[-3:]) / 3
            earlier_avg = sum(escalation_scores[:-3]) / len(escalation_scores[:-3])
            
            if recent_avg > earlier_avg + 0.2:
                return 'increasing'
            elif recent_avg < earlier_avg - 0.2:
                return 'decreasing'
        
        return 'stable'
    
    def _get_most_common(self, items: List) -> str:
        """Get most common item in list."""
        if not items:
            return 'neutral'
        return max(set(items), key=items.count) 
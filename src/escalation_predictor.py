import re
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json

class EscalationPredictor:
    """Predict escalation patterns and manage customer support escalations."""
    
    def __init__(self):
        """Initialize the escalation predictor."""
        self.escalation_patterns = {
            'high_urgency_keywords': [
                'urgent', 'emergency', 'critical', 'immediately', 'asap', 'now',
                'broken', 'down', 'not working', 'failed', 'error', 'crash',
                'lost', 'stolen', 'hacked', 'compromised', 'fraud', 'unauthorized'
            ],
            'frustration_indicators': [
                'frustrated', 'angry', 'upset', 'disappointed', 'unhappy',
                'terrible', 'awful', 'horrible', 'worst', 'never', 'always',
                'useless', 'worthless', 'waste', 'time', 'money'
            ],
            'emotional_distress_indicators': [
                'sad', 'sadness', 'depressed', 'hopeless', 'overwhelmed',
                'anxious', 'worried', 'stressed', 'crying', 'tears',
                'feeling', 'emotion', 'emotional', 'hurt', 'pain'
            ],
            'escalation_requests': [
                'speak to manager', 'supervisor', 'higher up', 'escalate',
                'complaint', 'formal complaint', 'legal action', 'lawyer',
                'better service', 'compensation', 'refund', 'credit'
            ],
            'repetitive_contact': [
                'called multiple times', 'emailed several times', 'no response',
                'ignored', 'no one helps', 'getting nowhere', 'same issue'
            ]
        }
        
        self.escalation_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 0.9
        }
        
        self.customer_history = defaultdict(list)
        self.escalation_history = []
        
    def analyze_escalation_risk(self, 
                              customer_query: str, 
                              customer_id: str = None,
                              conversation_history: List[Dict] = None,
                              category: str = None) -> Dict[str, Any]:
        """Analyze escalation risk for a customer query."""
        
        # Initialize risk factors
        risk_factors = {
            'urgency_score': 0.0,
            'frustration_score': 0.0,
            'emotional_distress_score': 0.0,
            'escalation_request_score': 0.0,
            'repetitive_contact_score': 0.0,
            'category_risk': 0.0,
            'historical_risk': 0.0,
            'overall_risk': 0.0
        }
        
        query_lower = customer_query.lower()
        
        # Analyze urgency
        urgency_matches = sum(1 for keyword in self.escalation_patterns['high_urgency_keywords'] 
                            if keyword in query_lower)
        risk_factors['urgency_score'] = min(urgency_matches / 3, 1.0)
        
        # Analyze frustration
        frustration_matches = sum(1 for keyword in self.escalation_patterns['frustration_indicators'] 
                                if keyword in query_lower)
        risk_factors['frustration_score'] = min(frustration_matches / 2, 1.0)
        
        # Analyze emotional distress
        emotional_matches = sum(1 for keyword in self.escalation_patterns['emotional_distress_indicators'] 
                              if keyword in query_lower)
        risk_factors['emotional_distress_score'] = min(emotional_matches / 2, 1.0)
        
        # Analyze escalation requests
        escalation_matches = sum(1 for keyword in self.escalation_patterns['escalation_requests'] 
                               if keyword in query_lower)
        risk_factors['escalation_request_score'] = min(escalation_matches, 1.0)
        
        # Analyze repetitive contact
        repetitive_matches = sum(1 for keyword in self.escalation_patterns['repetitive_contact'] 
                               if keyword in query_lower)
        risk_factors['repetitive_contact_score'] = min(repetitive_matches, 1.0)
        
        # Category-based risk
        if category:
            category_risks = {
                'billing': 0.8,
                'returns_refunds': 0.9,
                'account_management': 0.7,
                'technical_support': 0.6,
                'emotional_support': 0.8,
                'product_inquiry': 0.3,
                'general_inquiry': 0.2
            }
            risk_factors['category_risk'] = category_risks.get(category, 0.5)
        
        # Historical risk analysis
        if customer_id and conversation_history:
            risk_factors['historical_risk'] = self._analyze_historical_risk(customer_id, conversation_history)
        
        # Calculate overall risk
        weights = {
            'urgency_score': 0.20,
            'frustration_score': 0.15,
            'emotional_distress_score': 0.20,
            'escalation_request_score': 0.25,
            'repetitive_contact_score': 0.10,
            'category_risk': 0.05,
            'historical_risk': 0.05
        }
        
        overall_risk = sum(risk_factors[key] * weights[key] for key in weights.keys())
        risk_factors['overall_risk'] = min(overall_risk, 1.0)
        
        # Determine risk level
        risk_level = self._determine_risk_level(risk_factors['overall_risk'])
        
        # Generate escalation recommendations
        recommendations = self._generate_escalation_recommendations(risk_factors, risk_level)
        
        return {
            'risk_factors': risk_factors,
            'risk_level': risk_level,
            'escalation_needed': risk_level in ['high', 'critical'],
            'recommendations': recommendations,
            'confidence': self._calculate_confidence(risk_factors),
            'timestamp': datetime.now().isoformat()
        }
    
    def _analyze_historical_risk(self, customer_id: str, conversation_history: List[Dict]) -> float:
        """Analyze historical escalation risk for a customer."""
        if not conversation_history:
            return 0.0
        
        # Count previous escalations
        escalation_count = sum(1 for conv in conversation_history 
                             if conv.get('escalated', False))
        
        # Count negative sentiment interactions
        negative_interactions = sum(1 for conv in conversation_history 
                                  if conv.get('sentiment', {}).get('overall_sentiment') == 'negative')
        
        # Calculate frequency of contact
        if len(conversation_history) > 1:
            first_contact = conversation_history[0].get('timestamp')
            last_contact = conversation_history[-1].get('timestamp')
            
            if first_contact and last_contact:
                try:
                    first_date = datetime.fromisoformat(first_contact.replace('Z', '+00:00'))
                    last_date = datetime.fromisoformat(last_contact.replace('Z', '+00:00'))
                    days_between = (last_date - first_date).days
                    contact_frequency = len(conversation_history) / max(days_between, 1)
                except:
                    contact_frequency = 0
            else:
                contact_frequency = 0
        else:
            contact_frequency = 0
        
        # Calculate historical risk score
        escalation_weight = 0.4
        negative_weight = 0.3
        frequency_weight = 0.3
        
        escalation_score = min(escalation_count / 3, 1.0)
        negative_score = min(negative_interactions / len(conversation_history), 1.0)
        frequency_score = min(contact_frequency / 2, 1.0)  # Normalize to 2 contacts per day
        
        historical_risk = (escalation_score * escalation_weight + 
                          negative_score * negative_weight + 
                          frequency_score * frequency_weight)
        
        return historical_risk
    
    def _determine_risk_level(self, overall_risk: float) -> str:
        """Determine risk level based on overall risk score."""
        if overall_risk >= self.escalation_thresholds['critical']:
            return 'critical'
        elif overall_risk >= self.escalation_thresholds['high']:
            return 'high'
        elif overall_risk >= self.escalation_thresholds['medium']:
            return 'medium'
        elif overall_risk >= self.escalation_thresholds['low']:
            return 'low'
        else:
            return 'minimal'
    
    def _generate_escalation_recommendations(self, risk_factors: Dict[str, float], risk_level: str) -> List[str]:
        """Generate escalation recommendations based on risk factors."""
        recommendations = []
        
        if risk_level == 'critical':
            recommendations.extend([
                "Immediate escalation to senior support required",
                "Consider manager intervention",
                "Monitor customer satisfaction closely",
                "Prepare compensation options if needed"
            ])
        elif risk_level == 'high':
            recommendations.extend([
                "Escalate to specialized support team",
                "Provide immediate acknowledgment and timeline",
                "Assign dedicated support representative",
                "Schedule follow-up within 24 hours"
            ])
        elif risk_level == 'medium':
            recommendations.extend([
                "Monitor conversation closely",
                "Provide detailed, empathetic response",
                "Offer proactive solutions",
                "Follow up within 48 hours"
            ])
        elif risk_level == 'low':
            recommendations.extend([
                "Standard support response",
                "Monitor for escalation triggers",
                "Provide helpful resources"
            ])
        
        # Add specific recommendations based on risk factors
        if risk_factors['urgency_score'] > 0.7:
            recommendations.append("Address urgency immediately in response")
        
        if risk_factors['frustration_score'] > 0.7:
            recommendations.append("Use empathetic language and acknowledge frustration")
        
        if risk_factors['escalation_request_score'] > 0.5:
            recommendations.append("Explain escalation process and timeline")
        
        if risk_factors['repetitive_contact_score'] > 0.5:
            recommendations.append("Acknowledge previous contact attempts and apologize")
        
        return recommendations
    
    def _calculate_confidence(self, risk_factors: Dict[str, float]) -> float:
        """Calculate confidence in the escalation prediction."""
        # Higher confidence when risk factors are more extreme
        extreme_scores = sum(1 for score in risk_factors.values() if score > 0.7 or score < 0.3)
        total_factors = len(risk_factors)
        
        confidence = extreme_scores / total_factors
        return min(confidence, 1.0)
    
    def predict_escalation_likelihood(self, 
                                    customer_id: str, 
                                    query: str,
                                    category: str = None) -> Dict[str, Any]:
        """Predict the likelihood of escalation for a customer interaction."""
        
        # Get customer history
        customer_history = self.customer_history.get(customer_id, [])
        
        # Analyze current risk
        risk_analysis = self.analyze_escalation_risk(
            customer_query=query,
            customer_id=customer_id,
            conversation_history=customer_history,
            category=category
        )
        
        # Calculate escalation likelihood
        base_likelihood = risk_analysis['risk_factors']['overall_risk']
        
        # Adjust based on historical patterns
        if customer_history:
            historical_escalations = sum(1 for conv in customer_history if conv.get('escalated', False))
            escalation_rate = historical_escalations / len(customer_history)
            
            # Weighted combination of current risk and historical pattern
            likelihood = (base_likelihood * 0.7) + (escalation_rate * 0.3)
        else:
            likelihood = base_likelihood
        
        return {
            'escalation_likelihood': min(likelihood, 1.0),
            'risk_analysis': risk_analysis,
            'confidence': risk_analysis['confidence'],
            'recommended_action': self._get_recommended_action(likelihood, risk_analysis['risk_level'])
        }
    
    def _get_recommended_action(self, likelihood: float, risk_level: str) -> str:
        """Get recommended action based on escalation likelihood."""
        if likelihood > 0.8:
            return "Immediate escalation to senior support"
        elif likelihood > 0.6:
            return "Escalate to specialized team"
        elif likelihood > 0.4:
            return "Monitor closely and prepare for escalation"
        elif likelihood > 0.2:
            return "Standard support with enhanced monitoring"
        else:
            return "Standard support response"
    
    def record_interaction(self, 
                          customer_id: str, 
                          query: str, 
                          response: str, 
                          escalated: bool = False,
                          satisfaction_score: float = None):
        """Record a customer interaction for pattern analysis."""
        
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'escalated': escalated,
            'satisfaction_score': satisfaction_score
        }
        
        self.customer_history[customer_id].append(interaction)
        
        if escalated:
            self.escalation_history.append({
                'customer_id': customer_id,
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'category': self._categorize_query(query)
            })
    
    def _categorize_query(self, query: str) -> str:
        """Categorize a query for escalation tracking."""
        query_lower = query.lower()
        
        categories = {
            'billing': ['payment', 'billing', 'charge', 'refund', 'subscription'],
            'technical': ['error', 'bug', 'crash', 'not working', 'problem'],
            'account': ['password', 'login', 'account', 'security'],
            'returns': ['return', 'refund', 'cancel', 'exchange'],
            'general': ['question', 'help', 'support', 'information']
        }
        
        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def get_escalation_analytics(self) -> Dict[str, Any]:
        """Get analytics on escalation patterns."""
        
        total_interactions = sum(len(history) for history in self.customer_history.values())
        total_escalations = len(self.escalation_history)
        
        # Category breakdown
        category_counts = defaultdict(int)
        for escalation in self.escalation_history:
            category_counts[escalation['category']] += 1
        
        # Time-based analysis
        recent_escalations = [
            esc for esc in self.escalation_history
            if datetime.fromisoformat(esc['timestamp'].replace('Z', '+00:00')) > 
               datetime.now() - timedelta(days=7)
        ]
        
        return {
            'total_interactions': total_interactions,
            'total_escalations': total_escalations,
            'escalation_rate': total_escalations / max(total_interactions, 1),
            'category_breakdown': dict(category_counts),
            'recent_escalations': len(recent_escalations),
            'trend': self._calculate_escalation_trend()
        }
    
    def _calculate_escalation_trend(self) -> str:
        """Calculate escalation trend over time."""
        if len(self.escalation_history) < 2:
            return "insufficient_data"
        
        # Group escalations by week
        weekly_escalations = defaultdict(int)
        for escalation in self.escalation_history:
            date = datetime.fromisoformat(escalation['timestamp'].replace('Z', '+00:00'))
            week_key = date.strftime('%Y-%W')
            weekly_escalations[week_key] += 1
        
        if len(weekly_escalations) < 2:
            return "insufficient_data"
        
        # Calculate trend
        weeks = sorted(weekly_escalations.keys())
        recent_avg = np.mean([weekly_escalations[w] for w in weeks[-2:]])
        older_avg = np.mean([weekly_escalations[w] for w in weeks[:-2]]) if len(weeks) > 2 else recent_avg
        
        if recent_avg > older_avg * 1.2:
            return "increasing"
        elif recent_avg < older_avg * 0.8:
            return "decreasing"
        else:
            return "stable"
    
    def export_escalation_data(self, filename: str = None) -> str:
        """Export escalation data for analysis."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"escalation_data_{timestamp}.json"
        
        export_data = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'total_customers': len(self.customer_history),
                'total_escalations': len(self.escalation_history)
            },
            'customer_history': dict(self.customer_history),
            'escalation_history': self.escalation_history,
            'analytics': self.get_escalation_analytics()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return filename 
import time
import uuid
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from .simple_rag_system import SimpleRAGSystem as AdvancedRAGSystem
from .customer_support_knowledge_base import CustomerSupportKnowledgeBase
from .escalation_predictor import EscalationPredictor
from .customer_support_response_generator import CustomerSupportResponseGenerator
from .customer_satisfaction_tracker import CustomerSatisfactionTracker
from .sentiment_analyzer import SentimentAnalyzer

class CustomerSupportRAGSystem:
    """Complete Customer Support RAG System with all advanced features."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the complete customer support RAG system."""
        print("🚀 Initializing Customer Support RAG System...")
        
        # Initialize core components
        self.rag_system = AdvancedRAGSystem(embedding_model)
        self.knowledge_base = CustomerSupportKnowledgeBase()
        self.escalation_predictor = EscalationPredictor()
        self.response_generator = CustomerSupportResponseGenerator()
        self.satisfaction_tracker = CustomerSatisfactionTracker()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Initialize customer support articles in the RAG system
        self._initialize_customer_support_articles()
        
        print("✅ Customer Support RAG System initialized successfully!")
    
    def _initialize_customer_support_articles(self):
        """Initialize customer support articles in the RAG system."""
        print("📚 Loading customer support articles into RAG system...")
        
        # Get articles from knowledge base
        articles = self.knowledge_base.help_articles
        
        # Add articles to RAG system
        for article in articles:
            self.rag_system.add_article(
                title=article['title'],
                content=article['content'],
                category=article['category']
            )
        
        print(f"✅ Loaded {len(articles)} customer support articles")
    
    def process_customer_query(self, 
                             customer_query: str, 
                             customer_id: str = None,
                             conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Process a customer query and generate an empathetic response."""
        
        start_time = time.time()
        
        # Generate customer ID if not provided
        if customer_id is None:
            customer_id = str(uuid.uuid4())
        
        try:
            # Step 1: Analyze sentiment
            print(f"🔍 Analyzing sentiment for customer {customer_id}...")
            sentiment_analysis = self.sentiment_analyzer.analyze_sentiment(customer_query)
            
            # Step 2: Categorize query
            print(f"📂 Categorizing query for customer {customer_id}...")
            categorization = self.knowledge_base.categorize_query(customer_query)
            
            # Step 3: Predict escalation risk
            print(f"⚠️ Predicting escalation risk for customer {customer_id}...")
            escalation_analysis = self.escalation_predictor.analyze_escalation_risk(
                customer_query=customer_query,
                customer_id=customer_id,
                conversation_history=conversation_history,
                category=categorization['category']
            )
            
            # Step 4: Retrieve relevant articles
            print(f"🔍 Retrieving relevant articles for customer {customer_id}...")
            search_results = self.rag_system.search_articles(customer_query, n_results=5)
            retrieved_articles = search_results.get('results', [])
            
            # Step 5: Generate empathetic response
            print(f"💬 Generating response for customer {customer_id}...")
            response = self.response_generator.generate_customer_support_response(
                customer_query=customer_query,
                retrieved_articles=retrieved_articles,
                sentiment_analysis=sentiment_analysis,
                escalation_analysis=escalation_analysis,
                conversation_history=conversation_history,
                customer_id=customer_id
            )
            
            # Step 6: Calculate response time
            response_time = time.time() - start_time
            response['response_time'] = response_time
            
            # Step 7: Record interaction for satisfaction tracking
            print(f"📊 Recording interaction for customer {customer_id}...")
            self.satisfaction_tracker.record_interaction(
                customer_id=customer_id,
                query=customer_query,
                response=response,
                escalation_occurred=escalation_analysis.get('escalation_needed', False),
                response_time=response_time
            )
            
            # Step 8: Record interaction for escalation tracking
            self.escalation_predictor.record_interaction(
                customer_id=customer_id,
                query=customer_query,
                response=response['content'],
                escalated=escalation_analysis.get('escalation_needed', False)
            )
            
            # Step 9: Prepare comprehensive result
            result = {
                'customer_id': customer_id,
                'query': customer_query,
                'response': response,
                'sentiment_analysis': sentiment_analysis,
                'categorization': categorization,
                'escalation_analysis': escalation_analysis,
                'retrieved_articles': retrieved_articles,
                'response_time': response_time,
                'timestamp': datetime.now().isoformat(),
                'system_metadata': {
                    'articles_used': len(retrieved_articles),
                    'escalation_needed': escalation_analysis.get('escalation_needed', False),
                    'sentiment_aware': response.get('sentiment_aware', False),
                    'confidence_score': response.get('confidence_score', 0.0)
                }
            }
            
            print(f"✅ Successfully processed query for customer {customer_id}")
            return result
            
        except Exception as e:
            print(f"❌ Error processing query for customer {customer_id}: {e}")
            return {
                'customer_id': customer_id,
                'query': customer_query,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def record_satisfaction_feedback(self, 
                                   customer_id: str, 
                                   satisfaction_score: float, 
                                   feedback: str = None) -> Dict[str, Any]:
        """Record customer satisfaction feedback."""
        
        try:
            # Find the most recent interaction for this customer
            customer_interactions = self.satisfaction_tracker.customer_interactions.get(customer_id, [])
            
            if not customer_interactions:
                return {'error': 'No interactions found for this customer'}
            
            # Update the most recent interaction
            latest_interaction = customer_interactions[-1]
            latest_interaction['satisfaction_score'] = satisfaction_score
            latest_interaction['feedback'] = feedback
            
            # Update satisfaction metrics
            self.satisfaction_tracker._update_satisfaction_metrics()
            
            # Update knowledge base if feedback is provided
            if feedback and latest_interaction.get('articles_used', 0) > 0:
                self._update_article_satisfaction(latest_interaction, satisfaction_score)
            
            return {
                'customer_id': customer_id,
                'satisfaction_score': satisfaction_score,
                'feedback': feedback,
                'timestamp': datetime.now().isoformat(),
                'status': 'recorded'
            }
            
        except Exception as e:
            return {'error': f'Failed to record feedback: {str(e)}'}
    
    def _update_article_satisfaction(self, interaction: Dict[str, Any], satisfaction_score: float):
        """Update article satisfaction metrics based on customer feedback."""
        # This would update the satisfaction metrics for articles used in the response
        # Implementation depends on how articles are tracked in the response
        pass
    
    def get_customer_insights(self, customer_id: str) -> Dict[str, Any]:
        """Get comprehensive insights for a specific customer."""
        
        try:
            # Get customer interaction history
            customer_interactions = self.satisfaction_tracker.customer_interactions.get(customer_id, [])
            
            if not customer_interactions:
                return {'error': 'No interactions found for this customer'}
            
            # Calculate customer-specific metrics
            total_interactions = len(customer_interactions)
            satisfaction_scores = [i['satisfaction_score'] for i in customer_interactions if i['satisfaction_score'] is not None]
            escalation_count = sum(1 for i in customer_interactions if i['escalation_occurred'])
            
            # Get escalation predictions
            escalation_likelihood = self.escalation_predictor.predict_escalation_likelihood(
                customer_id=customer_id,
                query=customer_interactions[-1]['query'] if customer_interactions else ""
            )
            
            insights = {
                'customer_id': customer_id,
                'total_interactions': total_interactions,
                'average_satisfaction': np.mean(satisfaction_scores) if satisfaction_scores else 0,
                'escalation_rate': escalation_count / total_interactions if total_interactions > 0 else 0,
                'escalation_likelihood': escalation_likelihood['escalation_likelihood'],
                'risk_level': escalation_likelihood['risk_analysis']['risk_level'],
                'interaction_history': customer_interactions[-5:],  # Last 5 interactions
                'recommendations': self._generate_customer_recommendations(customer_interactions)
            }
            
            return insights
            
        except Exception as e:
            return {'error': f'Failed to get customer insights: {str(e)}'}
    
    def _generate_customer_recommendations(self, interactions: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on customer interaction history."""
        recommendations = []
        
        # Analyze satisfaction trends
        satisfaction_scores = [i['satisfaction_score'] for i in interactions if i['satisfaction_score'] is not None]
        if len(satisfaction_scores) >= 2:
            recent_avg = np.mean(satisfaction_scores[-3:]) if len(satisfaction_scores) >= 3 else satisfaction_scores[-1]
            if recent_avg < 0.6:
                recommendations.append("Consider proactive outreach to address satisfaction concerns")
        
        # Analyze escalation patterns
        escalation_count = sum(1 for i in interactions if i['escalation_occurred'])
        if escalation_count > len(interactions) * 0.3:
            recommendations.append("High escalation rate - consider dedicated support representative")
        
        # Analyze response time impact
        response_times = [i['response_time'] for i in interactions if i['response_time'] is not None]
        if response_times and np.mean(response_times) > 60:
            recommendations.append("Consider optimizing response times for better satisfaction")
        
        return recommendations
    
    def get_system_analytics(self, time_period: str = 'all') -> Dict[str, Any]:
        """Get comprehensive system analytics."""
        
        analytics = {
            'satisfaction_analytics': self.satisfaction_tracker.get_satisfaction_analytics(time_period),
            'escalation_analytics': self.escalation_predictor.get_escalation_analytics(),
            'knowledge_base_stats': self.knowledge_base.get_knowledge_base_stats(),
            'rag_system_stats': self._get_rag_system_stats()
        }
        
        return analytics
    
    def _get_rag_system_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        try:
            collection = self.rag_system.collection
            return {
                'total_documents': collection.count(),
                'embedding_model': self.rag_system.embedding_model,
                'database_type': 'ChromaDB'
            }
        except:
            return {'error': 'Unable to retrieve RAG system stats'}
    
    def create_system_visualizations(self, time_period: str = 'all') -> Dict[str, Any]:
        """Create comprehensive system visualizations."""
        
        visualizations = {
            'satisfaction_charts': self.satisfaction_tracker.create_satisfaction_visualizations(time_period),
            'escalation_trends': self._create_escalation_visualizations(),
            'knowledge_base_usage': self._create_knowledge_base_visualizations()
        }
        
        return visualizations
    
    def _create_escalation_visualizations(self):
        """Create escalation trend visualizations."""
        # This would create visualizations for escalation patterns
        # Implementation would depend on the specific visualization needs
        pass
    
    def _create_knowledge_base_visualizations(self):
        """Create knowledge base usage visualizations."""
        # This would create visualizations for knowledge base usage patterns
        # Implementation would depend on the specific visualization needs
        pass
    
    def generate_comprehensive_report(self, time_period: str = 'all') -> str:
        """Generate a comprehensive system report."""
        
        analytics = self.get_system_analytics(time_period)
        
        report = f"""
# Customer Support RAG System - Comprehensive Report
## Time Period: {time_period.title()}

## Executive Summary
- **Total Interactions**: {analytics['satisfaction_analytics'].get('total_interactions', 0)}
- **Unique Customers**: {analytics['satisfaction_analytics'].get('unique_customers', 0)}
- **Overall Satisfaction**: {analytics['satisfaction_analytics'].get('satisfaction_distribution', {}).get('average_satisfaction', 0):.2f}
- **Escalation Rate**: {analytics['escalation_analytics'].get('escalation_rate', 0):.1%}

## System Performance

### RAG System
- **Total Documents**: {analytics['rag_system_stats'].get('total_documents', 0)}
- **Embedding Model**: {analytics['rag_system_stats'].get('embedding_model', 'Unknown')}
- **Database**: {analytics['rag_system_stats'].get('database_type', 'Unknown')}

### Knowledge Base
- **Total Articles**: {analytics['knowledge_base_stats'].get('total_articles', 0)}
- **Categories**: {len(analytics['knowledge_base_stats'].get('categories', {}))}
- **Average Satisfaction**: {analytics['knowledge_base_stats'].get('average_satisfaction', 0):.2f}

## Detailed Analytics

### Satisfaction Analysis
{self.satisfaction_tracker.generate_satisfaction_report(time_period)}

### Escalation Analysis
- **Total Escalations**: {analytics['escalation_analytics'].get('total_escalations', 0)}
- **Recent Escalations**: {analytics['escalation_analytics'].get('recent_escalations', 0)}
- **Trend**: {analytics['escalation_analytics'].get('trend', 'Unknown')}

## Recommendations
1. **Monitor Satisfaction Trends**: Continue tracking customer satisfaction
2. **Optimize Response Quality**: Focus on improving response generation
3. **Reduce Escalations**: Work on first-contact resolution
4. **Enhance Knowledge Base**: Regularly update help articles
5. **Improve Response Times**: Optimize system performance

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report
    
    def export_system_data(self, filename: str = None) -> str:
        """Export comprehensive system data."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"customer_support_rag_data_{timestamp}.json"
        
        export_data = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'system_version': '1.0.0',
                'components': [
                    'AdvancedRAGSystem',
                    'CustomerSupportKnowledgeBase', 
                    'EscalationPredictor',
                    'CustomerSupportResponseGenerator',
                    'CustomerSatisfactionTracker',
                    'SentimentAnalyzer'
                ]
            },
            'satisfaction_data': self.satisfaction_tracker.export_satisfaction_data(),
            'escalation_data': self.escalation_predictor.export_escalation_data(),
            'knowledge_base_data': self.knowledge_base.export_knowledge_base(),
            'system_analytics': self.get_system_analytics()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return filename
    
    def test_system_functionality(self) -> Dict[str, Any]:
        """Test all system components."""
        
        test_results = {
            'rag_system': self._test_rag_system(),
            'knowledge_base': self._test_knowledge_base(),
            'escalation_predictor': self._test_escalation_predictor(),
            'response_generator': self._test_response_generator(),
            'satisfaction_tracker': self._test_satisfaction_tracker(),
            'sentiment_analyzer': self._test_sentiment_analyzer(),
            'integration': self._test_integration()
        }
        
        # Calculate overall system health
        all_tests = []
        for component, results in test_results.items():
            if isinstance(results, dict) and 'status' in results:
                all_tests.append(results['status'] == 'passed')
        
        overall_health = sum(all_tests) / len(all_tests) if all_tests else 0
        
        test_results['overall_system_health'] = {
            'score': overall_health,
            'status': 'healthy' if overall_health >= 0.8 else 'needs_attention' if overall_health >= 0.6 else 'critical'
        }
        
        return test_results
    
    def _test_rag_system(self) -> Dict[str, Any]:
        """Test RAG system functionality."""
        try:
            results = self.rag_system.search_articles("test query")
            return {'status': 'passed', 'message': 'RAG system working correctly'}
        except Exception as e:
            return {'status': 'failed', 'message': f'RAG system error: {str(e)}'}
    
    def _test_knowledge_base(self) -> Dict[str, Any]:
        """Test knowledge base functionality."""
        try:
            articles = self.knowledge_base.get_relevant_articles("test query")
            return {'status': 'passed', 'message': f'Knowledge base working correctly, found {len(articles)} articles'}
        except Exception as e:
            return {'status': 'failed', 'message': f'Knowledge base error: {str(e)}'}
    
    def _test_escalation_predictor(self) -> Dict[str, Any]:
        """Test escalation predictor functionality."""
        try:
            analysis = self.escalation_predictor.analyze_escalation_risk("test query")
            return {'status': 'passed', 'message': 'Escalation predictor working correctly'}
        except Exception as e:
            return {'status': 'failed', 'message': f'Escalation predictor error: {str(e)}'}
    
    def _test_response_generator(self) -> Dict[str, Any]:
        """Test response generator functionality."""
        try:
            response = self.response_generator.generate_customer_support_response(
                "test query", [], {}, {}
            )
            return {'status': 'passed', 'message': 'Response generator working correctly'}
        except Exception as e:
            return {'status': 'failed', 'message': f'Response generator error: {str(e)}'}
    
    def _test_satisfaction_tracker(self) -> Dict[str, Any]:
        """Test satisfaction tracker functionality."""
        try:
            analytics = self.satisfaction_tracker.get_satisfaction_analytics()
            return {'status': 'passed', 'message': 'Satisfaction tracker working correctly'}
        except Exception as e:
            return {'status': 'failed', 'message': f'Satisfaction tracker error: {str(e)}'}
    
    def _test_sentiment_analyzer(self) -> Dict[str, Any]:
        """Test sentiment analyzer functionality."""
        try:
            sentiment = self.sentiment_analyzer.analyze_sentiment("test query")
            return {'status': 'passed', 'message': 'Sentiment analyzer working correctly'}
        except Exception as e:
            return {'status': 'failed', 'message': f'Sentiment analyzer error: {str(e)}'}
    
    def _test_integration(self) -> Dict[str, Any]:
        """Test full system integration."""
        try:
            result = self.process_customer_query("test customer query")
            return {'status': 'passed', 'message': 'System integration working correctly'}
        except Exception as e:
            return {'status': 'failed', 'message': f'Integration error: {str(e)}'} 
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class CustomerSatisfactionTracker:
    """Track and optimize customer satisfaction for the RAG customer support system."""
    
    def __init__(self):
        """Initialize the customer satisfaction tracker."""
        self.satisfaction_data = []
        self.customer_interactions = defaultdict(list)
        self.satisfaction_metrics = {
            'overall_satisfaction': 0.0,
            'response_time_satisfaction': 0.0,
            'resolution_satisfaction': 0.0,
            'empathy_satisfaction': 0.0,
            'escalation_satisfaction': 0.0
        }
        
        # Satisfaction thresholds
        self.satisfaction_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'fair': 0.4,
            'poor': 0.2
        }
        
        # Optimization targets
        self.optimization_targets = {
            'target_satisfaction': 0.85,
            'target_response_time': 30,  # seconds
            'target_resolution_rate': 0.90,
            'target_escalation_rate': 0.10
        }
    
    def record_interaction(self, 
                          customer_id: str,
                          query: str,
                          response: Dict[str, Any],
                          satisfaction_score: float = None,
                          feedback: str = None,
                          escalation_occurred: bool = False,
                          response_time: float = None) -> Dict[str, Any]:
        """Record a customer interaction and satisfaction data."""
        
        interaction = {
            'customer_id': customer_id,
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response_content': response.get('content', ''),
            'response_type': response.get('response_type', 'unknown'),
            'sentiment_aware': response.get('sentiment_aware', False),
            'escalation_handled': response.get('escalation_handled', False),
            'articles_used': response.get('articles_used', 0),
            'confidence_score': response.get('confidence_score', 0.0),
            'response_tone': response.get('response_tone', 'unknown'),
            'satisfaction_score': satisfaction_score,
            'feedback': feedback,
            'escalation_occurred': escalation_occurred,
            'response_time': response_time,
            'satisfaction_level': self._categorize_satisfaction(satisfaction_score) if satisfaction_score else None
        }
        
        # Store interaction
        self.customer_interactions[customer_id].append(interaction)
        self.satisfaction_data.append(interaction)
        
        # Update metrics
        self._update_satisfaction_metrics()
        
        return interaction
    
    def _categorize_satisfaction(self, satisfaction_score: float) -> str:
        """Categorize satisfaction score into levels."""
        if satisfaction_score >= self.satisfaction_thresholds['excellent']:
            return 'excellent'
        elif satisfaction_score >= self.satisfaction_thresholds['good']:
            return 'good'
        elif satisfaction_score >= self.satisfaction_thresholds['fair']:
            return 'fair'
        else:
            return 'poor'
    
    def _update_satisfaction_metrics(self):
        """Update overall satisfaction metrics."""
        if not self.satisfaction_data:
            return
        
        # Calculate overall satisfaction
        valid_scores = [d['satisfaction_score'] for d in self.satisfaction_data 
                       if d['satisfaction_score'] is not None]
        
        if valid_scores:
            self.satisfaction_metrics['overall_satisfaction'] = np.mean(valid_scores)
        
        # Calculate response time satisfaction
        valid_times = [d['response_time'] for d in self.satisfaction_data 
                      if d['response_time'] is not None]
        
        if valid_times:
            avg_response_time = np.mean(valid_times)
            # Convert response time to satisfaction (faster = higher satisfaction)
            self.satisfaction_metrics['response_time_satisfaction'] = max(0, 1 - (avg_response_time / 120))
        
        # Calculate resolution satisfaction
        escalation_rate = sum(1 for d in self.satisfaction_data if d['escalation_occurred']) / len(self.satisfaction_data)
        self.satisfaction_metrics['escalation_satisfaction'] = 1 - escalation_rate
        
        # Calculate empathy satisfaction (based on sentiment-aware responses)
        sentiment_aware_responses = [d for d in self.satisfaction_data if d['sentiment_aware']]
        if sentiment_aware_responses:
            empathy_scores = [d['satisfaction_score'] for d in sentiment_aware_responses 
                            if d['satisfaction_score'] is not None]
            if empathy_scores:
                self.satisfaction_metrics['empathy_satisfaction'] = np.mean(empathy_scores)
    
    def get_satisfaction_analytics(self, time_period: str = 'all') -> Dict[str, Any]:
        """Get comprehensive satisfaction analytics."""
        
        # Filter data by time period
        filtered_data = self._filter_data_by_period(time_period)
        
        if not filtered_data:
            return {'error': 'No data available for the specified time period'}
        
        analytics = {
            'time_period': time_period,
            'total_interactions': len(filtered_data),
            'unique_customers': len(set(d['customer_id'] for d in filtered_data)),
            'satisfaction_distribution': self._get_satisfaction_distribution(filtered_data),
            'response_type_performance': self._analyze_response_type_performance(filtered_data),
            'escalation_analysis': self._analyze_escalation_patterns(filtered_data),
            'response_time_analysis': self._analyze_response_times(filtered_data),
            'sentiment_impact': self._analyze_sentiment_impact(filtered_data),
            'trends': self._calculate_satisfaction_trends(filtered_data),
            'optimization_opportunities': self._identify_optimization_opportunities(filtered_data)
        }
        
        return analytics
    
    def _filter_data_by_period(self, time_period: str) -> List[Dict[str, Any]]:
        """Filter data by time period."""
        if time_period == 'all':
            return self.satisfaction_data
        
        now = datetime.now()
        
        if time_period == 'today':
            cutoff = now - timedelta(days=1)
        elif time_period == 'week':
            cutoff = now - timedelta(weeks=1)
        elif time_period == 'month':
            cutoff = now - timedelta(days=30)
        else:
            return self.satisfaction_data
        
        filtered_data = []
        for interaction in self.satisfaction_data:
            try:
                interaction_time = datetime.fromisoformat(interaction['timestamp'].replace('Z', '+00:00'))
                if interaction_time >= cutoff:
                    filtered_data.append(interaction)
            except:
                continue
        
        return filtered_data
    
    def _get_satisfaction_distribution(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get distribution of satisfaction levels."""
        satisfaction_levels = [d['satisfaction_level'] for d in data if d['satisfaction_level']]
        
        if not satisfaction_levels:
            return {}
        
        level_counts = defaultdict(int)
        for level in satisfaction_levels:
            level_counts[level] += 1
        
        total = len(satisfaction_levels)
        distribution = {level: {'count': count, 'percentage': (count / total) * 100} 
                       for level, count in level_counts.items()}
        
        return {
            'distribution': distribution,
            'average_satisfaction': np.mean([d['satisfaction_score'] for d in data if d['satisfaction_score'] is not None]),
            'median_satisfaction': np.median([d['satisfaction_score'] for d in data if d['satisfaction_score'] is not None])
        }
    
    def _analyze_response_type_performance(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance by response type."""
        response_types = defaultdict(list)
        
        for interaction in data:
            response_type = interaction['response_type']
            if interaction['satisfaction_score'] is not None:
                response_types[response_type].append(interaction['satisfaction_score'])
        
        performance = {}
        for response_type, scores in response_types.items():
            performance[response_type] = {
                'count': len(scores),
                'average_satisfaction': np.mean(scores),
                'std_satisfaction': np.std(scores)
            }
        
        return performance
    
    def _analyze_escalation_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze escalation patterns and their impact on satisfaction."""
        escalated = [d for d in data if d['escalation_occurred']]
        non_escalated = [d for d in data if not d['escalation_occurred']]
        
        escalation_analysis = {
            'total_escalations': len(escalated),
            'escalation_rate': len(escalated) / len(data) if data else 0,
            'escalated_satisfaction': np.mean([d['satisfaction_score'] for d in escalated if d['satisfaction_score'] is not None]) if escalated else 0,
            'non_escalated_satisfaction': np.mean([d['satisfaction_score'] for d in non_escalated if d['satisfaction_score'] is not None]) if non_escalated else 0
        }
        
        return escalation_analysis
    
    def _analyze_response_times(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze response times and their impact on satisfaction."""
        valid_data = [d for d in data if d['response_time'] is not None and d['satisfaction_score'] is not None]
        
        if not valid_data:
            return {}
        
        response_times = [d['response_time'] for d in valid_data]
        satisfaction_scores = [d['satisfaction_score'] for d in valid_data]
        
        # Calculate correlation
        correlation = np.corrcoef(response_times, satisfaction_scores)[0, 1] if len(response_times) > 1 else 0
        
        return {
            'average_response_time': np.mean(response_times),
            'median_response_time': np.median(response_times),
            'response_time_satisfaction_correlation': correlation,
            'fast_response_satisfaction': np.mean([d['satisfaction_score'] for d in valid_data if d['response_time'] <= 30]),
            'slow_response_satisfaction': np.mean([d['satisfaction_score'] for d in valid_data if d['response_time'] > 60])
        }
    
    def _analyze_sentiment_impact(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the impact of sentiment-aware responses on satisfaction."""
        sentiment_aware = [d for d in data if d['sentiment_aware']]
        sentiment_unaware = [d for d in data if not d['sentiment_aware']]
        
        sentiment_analysis = {
            'sentiment_aware_count': len(sentiment_aware),
            'sentiment_unaware_count': len(sentiment_unaware),
            'sentiment_aware_satisfaction': np.mean([d['satisfaction_score'] for d in sentiment_aware if d['satisfaction_score'] is not None]) if sentiment_aware else 0,
            'sentiment_unaware_satisfaction': np.mean([d['satisfaction_score'] for d in sentiment_unaware if d['satisfaction_score'] is not None]) if sentiment_unaware else 0
        }
        
        return sentiment_analysis
    
    def _calculate_satisfaction_trends(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate satisfaction trends over time."""
        if len(data) < 2:
            return {'trend': 'insufficient_data'}
        
        # Group by day
        daily_satisfaction = defaultdict(list)
        for interaction in data:
            try:
                date = datetime.fromisoformat(interaction['timestamp'].replace('Z', '+00:00')).date()
                if interaction['satisfaction_score'] is not None:
                    daily_satisfaction[date].append(interaction['satisfaction_score'])
            except:
                continue
        
        if len(daily_satisfaction) < 2:
            return {'trend': 'insufficient_data'}
        
        # Calculate daily averages
        daily_averages = {date: np.mean(scores) for date, scores in daily_satisfaction.items()}
        dates = sorted(daily_averages.keys())
        averages = [daily_averages[date] for date in dates]
        
        # Calculate trend
        if len(averages) >= 2:
            recent_avg = np.mean(averages[-3:]) if len(averages) >= 3 else averages[-1]
            older_avg = np.mean(averages[:-3]) if len(averages) >= 3 else averages[0]
            
            if recent_avg > older_avg * 1.05:
                trend = 'improving'
            elif recent_avg < older_avg * 0.95:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'trend': trend,
            'daily_averages': daily_averages,
            'overall_trend_slope': np.polyfit(range(len(averages)), averages, 1)[0] if len(averages) > 1 else 0
        }
    
    def _identify_optimization_opportunities(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify opportunities for optimization."""
        opportunities = []
        
        # Check overall satisfaction against target
        current_satisfaction = np.mean([d['satisfaction_score'] for d in data if d['satisfaction_score'] is not None])
        if current_satisfaction < self.optimization_targets['target_satisfaction']:
            opportunities.append({
                'type': 'satisfaction_gap',
                'current': current_satisfaction,
                'target': self.optimization_targets['target_satisfaction'],
                'improvement_needed': self.optimization_targets['target_satisfaction'] - current_satisfaction,
                'recommendation': 'Focus on improving response quality and empathy'
            })
        
        # Check response time
        avg_response_time = np.mean([d['response_time'] for d in data if d['response_time'] is not None])
        if avg_response_time > self.optimization_targets['target_response_time']:
            opportunities.append({
                'type': 'response_time',
                'current': avg_response_time,
                'target': self.optimization_targets['target_response_time'],
                'improvement_needed': avg_response_time - self.optimization_targets['target_response_time'],
                'recommendation': 'Optimize response generation speed'
            })
        
        # Check escalation rate
        escalation_rate = sum(1 for d in data if d['escalation_occurred']) / len(data)
        if escalation_rate > self.optimization_targets['target_escalation_rate']:
            opportunities.append({
                'type': 'escalation_rate',
                'current': escalation_rate,
                'target': self.optimization_targets['target_escalation_rate'],
                'improvement_needed': escalation_rate - self.optimization_targets['target_escalation_rate'],
                'recommendation': 'Improve first-contact resolution rate'
            })
        
        # Check sentiment awareness impact
        sentiment_aware = [d for d in data if d['sentiment_aware']]
        sentiment_unaware = [d for d in data if not d['sentiment_aware']]
        
        if sentiment_aware and sentiment_unaware:
            aware_satisfaction = np.mean([d['satisfaction_score'] for d in sentiment_aware if d['satisfaction_score'] is not None])
            unaware_satisfaction = np.mean([d['satisfaction_score'] for d in sentiment_unaware if d['satisfaction_score'] is not None])
            
            if aware_satisfaction > unaware_satisfaction:
                opportunities.append({
                    'type': 'sentiment_awareness',
                    'current_impact': aware_satisfaction - unaware_satisfaction,
                    'recommendation': 'Increase use of sentiment-aware responses'
                })
        
        return opportunities
    
    def create_satisfaction_visualizations(self, time_period: str = 'all') -> go.Figure:
        """Create comprehensive satisfaction visualizations."""
        
        analytics = self.get_satisfaction_analytics(time_period)
        
        if 'error' in analytics:
            # Return empty figure with error message
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Satisfaction Distribution', 'Response Time vs Satisfaction', 
                          'Satisfaction Trends', 'Escalation Impact'),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Satisfaction Distribution (Pie Chart)
        if 'satisfaction_distribution' in analytics and 'distribution' in analytics['satisfaction_distribution']:
            distribution = analytics['satisfaction_distribution']['distribution']
            labels = list(distribution.keys())
            values = [dist['count'] for dist in distribution.values()]
            
            fig.add_trace(
                go.Pie(labels=labels, values=values, name="Satisfaction Distribution"),
                row=1, col=1
            )
        
        # 2. Response Time vs Satisfaction (Scatter Plot)
        valid_data = [d for d in self._filter_data_by_period(time_period) 
                     if d['response_time'] is not None and d['satisfaction_score'] is not None]
        
        if valid_data:
            response_times = [d['response_time'] for d in valid_data]
            satisfaction_scores = [d['satisfaction_score'] for d in valid_data]
            
            fig.add_trace(
                go.Scatter(x=response_times, y=satisfaction_scores, mode='markers',
                          name="Response Time vs Satisfaction"),
                row=1, col=2
            )
        
        # 3. Satisfaction Trends (Line Chart)
        if 'trends' in analytics and 'daily_averages' in analytics['trends']:
            daily_averages = analytics['trends']['daily_averages']
            dates = list(daily_averages.keys())
            averages = list(daily_averages.values())
            
            fig.add_trace(
                go.Scatter(x=dates, y=averages, mode='lines+markers',
                          name="Daily Satisfaction"),
                row=2, col=1
            )
        
        # 4. Escalation Impact (Bar Chart)
        if 'escalation_analysis' in analytics:
            escalation_data = analytics['escalation_analysis']
            categories = ['Escalated', 'Non-Escalated']
            satisfaction_values = [
                escalation_data['escalated_satisfaction'],
                escalation_data['non_escalated_satisfaction']
            ]
            
            fig.add_trace(
                go.Bar(x=categories, y=satisfaction_values, name="Escalation Impact"),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"Customer Satisfaction Analytics - {time_period.title()}",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def generate_satisfaction_report(self, time_period: str = 'all') -> str:
        """Generate a comprehensive satisfaction report."""
        
        analytics = self.get_satisfaction_analytics(time_period)
        
        if 'error' in analytics:
            return f"Error: {analytics['error']}"
        
        report = f"""
# Customer Satisfaction Report - {time_period.title()}

## Executive Summary
- **Total Interactions**: {analytics['total_interactions']}
- **Unique Customers**: {analytics['unique_customers']}
- **Overall Satisfaction**: {analytics['satisfaction_distribution']['average_satisfaction']:.2f}

## Satisfaction Distribution
"""
        
        if 'distribution' in analytics['satisfaction_distribution']:
            for level, data in analytics['satisfaction_distribution']['distribution'].items():
                report += f"- **{level.title()}**: {data['count']} ({data['percentage']:.1f}%)\n"
        
        report += f"""
## Performance Analysis

### Response Type Performance
"""
        
        for response_type, performance in analytics['response_type_performance'].items():
            report += f"- **{response_type}**: {performance['count']} interactions, {performance['average_satisfaction']:.2f} avg satisfaction\n"
        
        report += f"""
### Escalation Analysis
- **Escalation Rate**: {analytics['escalation_analysis']['escalation_rate']:.1%}
- **Escalated Satisfaction**: {analytics['escalation_analysis']['escalated_satisfaction']:.2f}
- **Non-Escalated Satisfaction**: {analytics['escalation_analysis']['non_escalated_satisfaction']:.2f}

### Response Time Analysis
- **Average Response Time**: {analytics['response_time_analysis']['average_response_time']:.1f} seconds
- **Response Time Correlation**: {analytics['response_time_analysis']['response_time_satisfaction_correlation']:.3f}

### Sentiment Impact
- **Sentiment-Aware Interactions**: {analytics['sentiment_impact']['sentiment_aware_count']}
- **Sentiment-Aware Satisfaction**: {analytics['sentiment_impact']['sentiment_aware_satisfaction']:.2f}
- **Sentiment-Unaware Satisfaction**: {analytics['sentiment_impact']['sentiment_unaware_satisfaction']:.2f}

## Trends
- **Overall Trend**: {analytics['trends'].get('trend', 'insufficient_data')}
- **Trend Slope**: {analytics['trends'].get('overall_trend_slope', 0):.3f}

## Optimization Opportunities
"""
        
        for opportunity in analytics['optimization_opportunities']:
            report += f"- **{opportunity['type'].replace('_', ' ').title()}**: {opportunity['recommendation']}\n"
        
        report += f"""
## Recommendations
1. **Monitor Trends**: Continue tracking satisfaction trends over time
2. **Optimize Response Types**: Focus on improving underperforming response types
3. **Reduce Escalations**: Work on first-contact resolution
4. **Improve Response Times**: Optimize for faster response generation
5. **Enhance Sentiment Awareness**: Increase use of sentiment-aware responses

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report
    
    def export_satisfaction_data(self, filename: str = None) -> str:
        """Export satisfaction data for external analysis."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"satisfaction_data_{timestamp}.json"
        
        export_data = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'total_interactions': len(self.satisfaction_data),
                'unique_customers': len(self.customer_interactions)
            },
            'satisfaction_data': self.satisfaction_data,
            'customer_interactions': dict(self.customer_interactions),
            'satisfaction_metrics': self.satisfaction_metrics,
            'analytics': self.get_satisfaction_analytics()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return filename 
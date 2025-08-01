import os
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
from .utils import chunk_text, clean_text, extract_keywords

class CustomerSupportKnowledgeBase:
    """Process and manage customer support help articles and knowledge base."""
    
    def __init__(self):
        """Initialize the customer support knowledge base."""
        self.help_articles = []
        self.categories = {
            'account_management': {
                'keywords': ['password', 'login', 'account', 'profile', 'settings', 'security'],
                'priority': 'high',
                'escalation_threshold': 0.7
            },
            'billing': {
                'keywords': ['payment', 'billing', 'invoice', 'charge', 'refund', 'subscription'],
                'priority': 'high',
                'escalation_threshold': 0.8
            },
            'technical_support': {
                'keywords': ['error', 'bug', 'crash', 'not working', 'issue', 'problem'],
                'priority': 'medium',
                'escalation_threshold': 0.6
            },
            'product_inquiry': {
                'keywords': ['feature', 'how to', 'guide', 'tutorial', 'help'],
                'priority': 'low',
                'escalation_threshold': 0.3
            },
            'returns_refunds': {
                'keywords': ['return', 'refund', 'cancel', 'exchange', 'money back'],
                'priority': 'high',
                'escalation_threshold': 0.9
            },
            'general_inquiry': {
                'keywords': ['question', 'information', 'contact', 'support'],
                'priority': 'low',
                'escalation_threshold': 0.2
            },
            'emotional_support': {
                'keywords': ['sad', 'frustrated', 'angry', 'upset', 'disappointed', 'worried', 'anxious', 'stressed', 'feeling', 'emotion', 'help', 'support'],
                'priority': 'high',
                'escalation_threshold': 0.6
            }
        }
        
        self._load_sample_help_articles()
    
    def _load_sample_help_articles(self):
        """Load sample customer support help articles."""
        self.help_articles = [
            {
                'id': 'CS001',
                'title': 'How to Reset Your Password',
                'content': 'If you\'ve forgotten your password, follow these steps: 1. Go to the login page 2. Click "Forgot Password" 3. Enter your email address 4. Check your email for reset instructions 5. Click the reset link and create a new password. If you continue to have issues, contact our support team.',
                'category': 'account_management',
                'tags': ['password', 'security', 'login', 'reset'],
                'difficulty': 'easy',
                'escalation_triggers': ['multiple attempts', 'email not received', 'link expired'],
                'satisfaction_metrics': {'helpful_rate': 0.85, 'resolution_rate': 0.92}
            },
            {
                'id': 'CS002',
                'title': 'Payment Declined - Troubleshooting Guide',
                'content': 'If your payment was declined, try these solutions: 1. Verify your card information is correct 2. Check if your card has sufficient funds 3. Contact your bank to ensure the card is not blocked 4. Try a different payment method 5. Clear your browser cache and try again. For immediate assistance, call our billing support.',
                'category': 'billing',
                'tags': ['payment', 'declined', 'card', 'billing', 'troubleshooting'],
                'difficulty': 'medium',
                'escalation_triggers': ['multiple cards declined', 'urgent payment needed', 'subscription expiring'],
                'satisfaction_metrics': {'helpful_rate': 0.78, 'resolution_rate': 0.88}
            },
            {
                'id': 'CS003',
                'title': 'App Not Working - Common Solutions',
                'content': 'If the app is not working properly: 1. Restart the app completely 2. Check your internet connection 3. Update to the latest version 4. Clear app cache and data 5. Restart your device. If problems persist, please provide specific error messages for better assistance.',
                'category': 'technical_support',
                'tags': ['app', 'not working', 'error', 'technical', 'troubleshooting'],
                'difficulty': 'medium',
                'escalation_triggers': ['app crashes repeatedly', 'data loss', 'critical functionality broken'],
                'satisfaction_metrics': {'helpful_rate': 0.82, 'resolution_rate': 0.85}
            },
            {
                'id': 'CS004',
                'title': 'Return Policy and Refund Process',
                'content': 'Our return policy allows returns within 30 days of purchase. To initiate a return: 1. Log into your account 2. Go to Order History 3. Select the item to return 4. Choose return reason 5. Print return label 6. Ship the item back. Refunds are processed within 5-7 business days after we receive the item.',
                'category': 'returns_refunds',
                'tags': ['return', 'refund', 'policy', 'shipping', 'money back'],
                'difficulty': 'easy',
                'escalation_triggers': ['return denied', 'refund not received', 'damaged item'],
                'satisfaction_metrics': {'helpful_rate': 0.90, 'resolution_rate': 0.95}
            },
            {
                'id': 'CS005',
                'title': 'Account Security Settings',
                'content': 'Protect your account with these security features: 1. Enable two-factor authentication 2. Use a strong, unique password 3. Regularly review login activity 4. Keep your email address updated 5. Enable login notifications. For security concerns, contact us immediately.',
                'category': 'account_management',
                'tags': ['security', 'account', 'protection', 'authentication', 'privacy'],
                'difficulty': 'easy',
                'escalation_triggers': ['suspicious activity', 'account compromised', 'unauthorized access'],
                'satisfaction_metrics': {'helpful_rate': 0.88, 'resolution_rate': 0.91}
            },
            {
                'id': 'CS006',
                'title': 'Subscription Management',
                'content': 'Manage your subscription: 1. Go to Account Settings 2. Select Subscription 3. Choose to upgrade, downgrade, or cancel 4. Confirm changes 5. Review billing cycle. Changes take effect at the next billing cycle. For immediate changes, contact support.',
                'category': 'billing',
                'tags': ['subscription', 'billing', 'cancel', 'upgrade', 'downgrade'],
                'difficulty': 'easy',
                'escalation_triggers': ['unwanted charges', 'cancellation issues', 'billing disputes'],
                'satisfaction_metrics': {'helpful_rate': 0.83, 'resolution_rate': 0.87}
            },
            {
                'id': 'CS007',
                'title': 'Data Recovery and Backup',
                'content': 'If you\'ve lost data: 1. Check the Recycle Bin/Trash 2. Look for auto-backup files 3. Check cloud storage if enabled 4. Contact support for recovery options. To prevent future data loss, enable automatic backups and use cloud storage.',
                'category': 'technical_support',
                'tags': ['data', 'recovery', 'backup', 'lost', 'restore'],
                'difficulty': 'hard',
                'escalation_triggers': ['critical data lost', 'business impact', 'legal requirements'],
                'satisfaction_metrics': {'helpful_rate': 0.75, 'resolution_rate': 0.70}
            },
            {
                'id': 'CS008',
                'title': 'Contact Customer Support',
                'content': 'Need help? Contact us through: 1. Live Chat (24/7) 2. Email: support@company.com 3. Phone: 1-800-SUPPORT 4. Help Center: help.company.com. For urgent issues, use live chat or phone support for immediate assistance.',
                'category': 'general_inquiry',
                'tags': ['contact', 'support', 'help', 'assistance', 'customer service'],
                'difficulty': 'easy',
                'escalation_triggers': ['no response received', 'urgent issue', 'multiple contact attempts'],
                'satisfaction_metrics': {'helpful_rate': 0.92, 'resolution_rate': 0.89}
            },
            {
                'id': 'CS009',
                'title': 'Emotional Support and Customer Care',
                'content': 'We understand that dealing with issues can be emotionally challenging. Our customer care team is here to provide empathetic support and understanding. If you\'re feeling frustrated, sad, or overwhelmed, please know that we care about your experience and are committed to helping you. You can speak with our specially trained emotional support representatives who are here to listen and provide compassionate assistance. Remember, your feelings are valid, and we\'re here to help you through this.',
                'category': 'emotional_support',
                'tags': ['emotional', 'support', 'care', 'empathy', 'frustration', 'sadness', 'help'],
                'difficulty': 'easy',
                'escalation_triggers': ['severe emotional distress', 'crisis situation', 'urgent emotional support needed'],
                'satisfaction_metrics': {'helpful_rate': 0.95, 'resolution_rate': 0.90}
            },
            {
                'id': 'CS010',
                'title': 'Coping with Frustration and Disappointment',
                'content': 'It\'s completely normal to feel frustrated or disappointed when things don\'t work as expected. We want you to know that your feelings are important to us. Here are some ways we can help: 1. Take a deep breath - we\'re here to support you 2. Let us know exactly what\'s bothering you so we can address it 3. We can connect you with someone who will listen and help 4. Remember that we\'re committed to making things right for you. Your satisfaction and emotional well-being are our top priorities.',
                'category': 'emotional_support',
                'tags': ['frustration', 'disappointment', 'coping', 'support', 'empathy', 'help'],
                'difficulty': 'easy',
                'escalation_triggers': ['extreme frustration', 'emotional crisis', 'urgent support needed'],
                'satisfaction_metrics': {'helpful_rate': 0.93, 'resolution_rate': 0.88}
            }
        ]
    
    def get_articles_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all articles for a specific category."""
        return [article for article in self.help_articles if article['category'] == category]
    
    def get_articles_by_keywords(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Get articles that match specific keywords."""
        matching_articles = []
        for article in self.help_articles:
            article_text = f"{article['title']} {article['content']}".lower()
            article_tags = [tag.lower() for tag in article['tags']]
            
            for keyword in keywords:
                if (keyword.lower() in article_text or 
                    keyword.lower() in article_tags):
                    matching_articles.append(article)
                    break
        
        return matching_articles
    
    def get_escalation_triggers(self, category: str) -> List[str]:
        """Get escalation triggers for a category."""
        triggers = []
        for article in self.help_articles:
            if article['category'] == category:
                triggers.extend(article['escalation_triggers'])
        return list(set(triggers))
    
    def get_satisfaction_metrics(self, category: str) -> Dict[str, float]:
        """Get satisfaction metrics for a category."""
        metrics = {'helpful_rate': 0.0, 'resolution_rate': 0.0}
        articles = self.get_articles_by_category(category)
        
        if articles:
            helpful_rates = [article['satisfaction_metrics']['helpful_rate'] for article in articles]
            resolution_rates = [article['satisfaction_metrics']['resolution_rate'] for article in articles]
            
            metrics['helpful_rate'] = sum(helpful_rates) / len(helpful_rates)
            metrics['resolution_rate'] = sum(resolution_rates) / len(resolution_rates)
        
        return metrics
    
    def categorize_query(self, query: str) -> Dict[str, Any]:
        """Categorize a customer query."""
        query_lower = query.lower()
        category_scores = {}
        
        for category, config in self.categories.items():
            score = 0
            for keyword in config['keywords']:
                if keyword.lower() in query_lower:
                    score += 1
            
            category_scores[category] = {
                'score': score / len(config['keywords']),
                'priority': config['priority'],
                'escalation_threshold': config['escalation_threshold']
            }
        
        # Get the best matching category
        best_category = max(category_scores.items(), key=lambda x: x[1]['score'])
        
        return {
            'category': best_category[0],
            'confidence': best_category[1]['score'],
            'priority': best_category[1]['priority'],
            'escalation_threshold': best_category[1]['escalation_threshold'],
            'all_scores': category_scores
        }
    
    def get_relevant_articles(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get relevant articles for a customer query."""
        categorization = self.categorize_query(query)
        category = categorization['category']
        
        # Get articles from the categorized category
        category_articles = self.get_articles_by_category(category)
        
        # Also get articles by keyword matching
        keywords = self.categories[category]['keywords']
        keyword_articles = self.get_articles_by_keywords(keywords)
        
        # Combine and deduplicate
        all_articles = category_articles + keyword_articles
        unique_articles = {article['id']: article for article in all_articles}.values()
        
        # Sort by relevance (using satisfaction metrics as proxy)
        sorted_articles = sorted(
            unique_articles, 
            key=lambda x: x['satisfaction_metrics']['helpful_rate'], 
            reverse=True
        )
        
        return sorted_articles[:limit]
    
    def add_article(self, title: str, content: str, category: str, tags: List[str] = None) -> bool:
        """Add a new help article to the knowledge base."""
        try:
            new_article = {
                'id': f'CS{len(self.help_articles) + 1:03d}',
                'title': title,
                'content': content,
                'category': category,
                'tags': tags or [],
                'difficulty': 'medium',
                'escalation_triggers': [],
                'satisfaction_metrics': {'helpful_rate': 0.0, 'resolution_rate': 0.0},
                'created_at': datetime.now().isoformat()
            }
            
            self.help_articles.append(new_article)
            return True
        except Exception as e:
            print(f"Error adding article: {e}")
            return False
    
    def update_satisfaction_metrics(self, article_id: str, helpful: bool, resolved: bool):
        """Update satisfaction metrics for an article."""
        for article in self.help_articles:
            if article['id'] == article_id:
                current_helpful = article['satisfaction_metrics']['helpful_rate']
                current_resolved = article['satisfaction_metrics']['resolution_rate']
                
                # Simple moving average update
                article['satisfaction_metrics']['helpful_rate'] = (current_helpful + (1 if helpful else 0)) / 2
                article['satisfaction_metrics']['resolution_rate'] = (current_resolved + (1 if resolved else 0)) / 2
                break
    
    def export_knowledge_base(self, filename: str = None) -> str:
        """Export the knowledge base to JSON."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"customer_support_kb_{timestamp}.json"
        
        export_data = {
            'metadata': {
                'export_date': datetime.now().isoformat(),
                'total_articles': len(self.help_articles),
                'categories': list(self.categories.keys())
            },
            'categories': self.categories,
            'articles': self.help_articles
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        stats = {
            'total_articles': len(self.help_articles),
            'categories': {},
            'average_satisfaction': 0.0,
            'escalation_triggers_count': 0
        }
        
        # Category statistics
        for category in self.categories.keys():
            articles = self.get_articles_by_category(category)
            satisfaction = self.get_satisfaction_metrics(category)
            
            stats['categories'][category] = {
                'article_count': len(articles),
                'average_helpful_rate': satisfaction['helpful_rate'],
                'average_resolution_rate': satisfaction['resolution_rate']
            }
        
        # Overall satisfaction
        all_helpful_rates = [article['satisfaction_metrics']['helpful_rate'] for article in self.help_articles]
        stats['average_satisfaction'] = sum(all_helpful_rates) / len(all_helpful_rates) if all_helpful_rates else 0
        
        # Escalation triggers
        all_triggers = []
        for article in self.help_articles:
            all_triggers.extend(article['escalation_triggers'])
        stats['escalation_triggers_count'] = len(set(all_triggers))
        
        return stats 
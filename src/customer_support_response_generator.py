import re
import random
from typing import List, Dict, Any, Optional
from datetime import datetime
from .utils import get_environment_variables, validate_api_key

class CustomerSupportResponseGenerator:
    """Generate empathetic and effective customer support responses."""
    
    def __init__(self):
        """Initialize the customer support response generator."""
        self.env_vars = get_environment_variables()
        self.openai_api_key = self.env_vars.get("OPENAI_API_KEY", "")
        
        # Comprehensive emotional response templates
        self.empathy_templates = {
            'joy_happiness': {
                'acknowledgment': [
                    "That's wonderful to hear! I'm so glad you're feeling happy and joyful.",
                    "I love hearing that you're happy! That's such a beautiful feeling.",
                    "That's fantastic! It's always wonderful when someone is feeling good and content.",
                    "I'm delighted to hear you're happy! That really brightens my day."
                ],
                'tone': 'enthusiastic and celebratory',
                'urgency_level': 'low'
            },
            'sadness_depression': {
                'acknowledgment': [
                    "I can see you're going through a difficult time, and I want you to know that your feelings are completely valid.",
                    "It's okay to feel sad, and you don't have to go through this alone. I'm here to listen and help.",
                    "I understand this is really hard for you right now, and I want you to know that we care about your well-being.",
                    "Your feelings matter, and I want to help you through this challenging time."
                ],
                'tone': 'compassionate and supportive',
                'urgency_level': 'high'
            },
            'anger_frustration': {
                'acknowledgment': [
                    "I completely understand why you're feeling frustrated, and I want to help resolve this for you.",
                    "I can see this situation has really upset you, and I'm here to help make things right.",
                    "I hear your frustration, and I want to assure you that we're going to work through this together.",
                    "I understand this has been a difficult experience, and I'm committed to finding a solution."
                ],
                'tone': 'understanding and solution-focused',
                'urgency_level': 'high'
            },
            'fear_anxiety': {
                'acknowledgment': [
                    "I can see you're feeling anxious, and I want you to know that it's completely normal to feel this way.",
                    "I understand this is worrying you, and I'm here to help you feel more secure and supported.",
                    "It's okay to feel afraid, and I want to help you through this. You're not alone.",
                    "I can sense your anxiety, and I want to help you find some peace and reassurance."
                ],
                'tone': 'calming and reassuring',
                'urgency_level': 'medium'
            },
            'surprise_shock': {
                'acknowledgment': [
                    "Wow, that sounds surprising! I can understand why you'd be shocked by that.",
                    "That's quite unexpected! I can see why you're feeling taken aback.",
                    "I can imagine how surprising that must be for you. How are you feeling about it?",
                    "That's definitely something that would catch anyone off guard. I'm here to help you process this."
                ],
                'tone': 'supportive and understanding',
                'urgency_level': 'low'
            },
            'disgust_contempt': {
                'acknowledgment': [
                    "I can understand why you'd feel that way about this situation.",
                    "That sounds really unpleasant, and I can see why you'd be upset about it.",
                    "I understand your reaction, and I want to help you address this concern.",
                    "I can see why this would bother you, and I'm here to help resolve it."
                ],
                'tone': 'validating and solution-oriented',
                'urgency_level': 'medium'
            },
            'love_affection': {
                'acknowledgment': [
                    "That's beautiful! It's wonderful to hear about love and connection in your life.",
                    "I love hearing about the love and care in your life! That's truly special.",
                    "That's heartwarming! It's always wonderful to hear about loving relationships.",
                    "I'm so glad you're experiencing such love and affection! That's truly precious."
                ],
                'tone': 'warm and celebratory',
                'urgency_level': 'low'
            },
            'pride_confidence': {
                'acknowledgment': [
                    "That's amazing! You should be proud of yourself for that accomplishment.",
                    "I'm so impressed by your confidence and what you've achieved!",
                    "That's fantastic! Your pride is completely justified, and I'm happy for you.",
                    "You have every right to be proud! That's a wonderful achievement."
                ],
                'tone': 'enthusiastic and encouraging',
                'urgency_level': 'low'
            },
            'shame_embarrassment': {
                'acknowledgment': [
                    "I want you to know that everyone makes mistakes, and it's okay to feel embarrassed.",
                    "I understand you're feeling ashamed, but please know that you're not alone in this.",
                    "It's completely normal to feel embarrassed sometimes, and I'm here to support you.",
                    "I can see you're feeling bad about this, but I want you to know that you're still valued and cared for."
                ],
                'tone': 'gentle and reassuring',
                'urgency_level': 'medium'
            },
            'jealousy_envy': {
                'acknowledgment': [
                    "I understand that feeling, and it's completely normal to experience jealousy sometimes.",
                    "I can see why you'd feel that way, and I want to help you work through these feelings.",
                    "Jealousy is a natural emotion, and I'm here to help you process it in a healthy way.",
                    "I understand these feelings, and I want to help you find peace with this situation."
                ],
                'tone': 'understanding and non-judgmental',
                'urgency_level': 'medium'
            },
            'hope_optimism': {
                'acknowledgment': [
                    "That's wonderful! Your hope and optimism are truly inspiring.",
                    "I love your positive outlook! It's beautiful to see such hope and determination.",
                    "Your optimism is amazing! I'm so glad you're feeling hopeful about the future.",
                    "That's fantastic! Your hope and positive attitude will help you achieve great things."
                ],
                'tone': 'encouraging and supportive',
                'urgency_level': 'low'
            },
            'gratitude_appreciation': {
                'acknowledgment': [
                    "That's beautiful! Your gratitude and appreciation are truly heartwarming.",
                    "I love hearing about your gratitude! It's wonderful to see such appreciation.",
                    "Your thankfulness is so touching! I'm glad you're feeling grateful and blessed.",
                    "That's wonderful! Your gratitude shows such a beautiful perspective on life."
                ],
                'tone': 'warm and appreciative',
                'urgency_level': 'low'
            },
            'curiosity_interest': {
                'acknowledgment': [
                    "That's great! Your curiosity and interest are wonderful qualities.",
                    "I love your enthusiasm for learning! Your curiosity is truly admirable.",
                    "That's fantastic! Your interest and engagement are wonderful to see.",
                    "I'm so glad you're curious about this! Your interest shows great engagement."
                ],
                'tone': 'enthusiastic and encouraging',
                'urgency_level': 'low'
            },
            'boredom_apathy': {
                'acknowledgment': [
                    "I understand feeling bored or disconnected, and I want to help you find something engaging.",
                    "I can see you're feeling unmotivated, and I'm here to help you find some excitement.",
                    "It's okay to feel bored sometimes, and I want to help you discover something interesting.",
                    "I understand this feeling, and I'm here to help you find meaning and engagement."
                ],
                'tone': 'encouraging and motivating',
                'urgency_level': 'medium'
            },
            'excitement_enthusiasm': {
                'acknowledgment': [
                    "That's fantastic! Your excitement is contagious and wonderful to see.",
                    "I love your enthusiasm! It's amazing to see such energy and passion.",
                    "That's incredible! Your excitement and energy are truly inspiring.",
                    "I'm so excited for you! Your enthusiasm is wonderful and well-deserved."
                ],
                'tone': 'energetic and celebratory',
                'urgency_level': 'low'
            },
            'calm_peace': {
                'acknowledgment': [
                    "That's beautiful! Your sense of calm and peace is truly wonderful.",
                    "I love hearing about your peaceful state! It's so refreshing and inspiring.",
                    "That's amazing! Your calm and serenity are truly admirable qualities.",
                    "I'm so glad you're feeling peaceful! That's a beautiful state to be in."
                ],
                'tone': 'gentle and peaceful',
                'urgency_level': 'low'
            },
            'emotional_support': {
                'acknowledgment': [
                    "I can see you're going through a difficult time, and I want you to know that your feelings are completely valid.",
                    "It's okay to feel this way, and you don't have to go through it alone. I'm here to listen and help.",
                    "I understand this is really hard for you right now, and I want you to know that we care about your well-being.",
                    "Your feelings matter, and I want to help you through this challenging time."
                ],
                'tone': 'compassionate and supportive',
                'urgency_level': 'high'
            },
            'high_frustration': {
                'acknowledgment': [
                    "I completely understand how frustrating this situation must be for you.",
                    "I can see why you're feeling this way, and I want you to know that I'm here to help.",
                    "I hear your frustration, and I want to assure you that we're going to resolve this together.",
                    "I understand this has been a difficult experience, and I'm committed to making it right."
                ],
                'tone': 'empathetic and understanding',
                'urgency_level': 'high'
            },
            'medium_frustration': {
                'acknowledgment': [
                    "I understand this is important to you, and I want to help resolve it.",
                    "I can see this has been challenging, and I'm here to assist you.",
                    "I understand your concern, and let me help you with this.",
                    "I want to make sure we get this sorted out for you right away."
                ],
                'tone': 'helpful and supportive',
                'urgency_level': 'medium'
            },
            'low_frustration': {
                'acknowledgment': [
                    "I'd be happy to help you with that.",
                    "Let me assist you with this right away.",
                    "I can help you with that, no problem.",
                    "I'll make sure we get this resolved for you."
                ],
                'tone': 'professional and helpful',
                'urgency_level': 'low'
            }
        }
        
        # Escalation response templates
        self.escalation_responses = {
            'critical': {
                'immediate': "I understand this is urgent and I want to help you resolve this immediately. Let me connect you with our senior support team right away.",
                'follow_up': "I've escalated your case and a senior representative will contact you within the next 30 minutes.",
                'tone': 'urgent and concerned'
            },
            'high': {
                'immediate': "I can see this is important to you. Let me work on finding the best solution for your situation.",
                'follow_up': "I'm actively working on your case and will provide updates as soon as I have more information.",
                'tone': 'attentive and responsive'
            },
            'medium': {
                'immediate': "I'll help you with this right away and make sure we get it resolved.",
                'follow_up': "I'm here to assist you with any questions you might have.",
                'tone': 'helpful and proactive'
            },
            'low': {
                'immediate': "I'll help you with this right away.",
                'follow_up': "I'm here to assist you with any questions you might have.",
                'tone': 'professional and friendly'
            }
        }
        
        # Customer satisfaction phrases
        self.satisfaction_phrases = {
            'positive': [
                "I'm glad I could help!",
                "I'm happy to assist you with this!",
                "That's wonderful to hear!",
                "I'm so glad we could resolve this for you!"
            ],
            'neutral': [
                "I hope this helps resolve your issue.",
                "Please let me know if you need any further assistance.",
                "I'm here if you have any other questions.",
                "Feel free to reach out if you need anything else."
            ],
            'negative': [
                "I understand this may not be the solution you were looking for.",
                "I want to make sure we find the right solution for you.",
                "Let me know if there's anything else I can do to help.",
                "I'm committed to finding the best resolution for you."
            ]
        }
    
    def generate_customer_support_response(self, 
                                         customer_query: str,
                                         retrieved_articles: List[Dict[str, Any]],
                                         sentiment_analysis: Dict[str, Any],
                                         escalation_analysis: Dict[str, Any],
                                         conversation_history: List[Dict] = None,
                                         customer_id: str = None) -> Dict[str, Any]:
        """Generate empathetic customer support response."""
        
        # Determine response tone based on sentiment and escalation risk
        sentiment = sentiment_analysis.get('overall_sentiment', 'neutral')
        escalation_risk = escalation_analysis.get('risk_level', 'low')
        
        # Get relevant context from articles
        context_text = self._format_article_context(retrieved_articles)
        
        # Generate response using appropriate method
        if self.openai_api_key and validate_api_key(self.openai_api_key):
            response = self._generate_ai_customer_response(
                customer_query, context_text, sentiment_analysis, 
                escalation_analysis, conversation_history
            )
        else:
            response = self._generate_template_customer_response(
                customer_query, context_text, sentiment_analysis, escalation_analysis
            )
        
        # Add escalation handling if needed
        if escalation_analysis.get('escalation_needed', False):
            escalation_response = self._add_escalation_handling(escalation_risk, escalation_analysis)
            response['content'] = escalation_response + "\n\n" + response['content']
        
        # Add customer satisfaction tracking
        response['satisfaction_tracking'] = self._generate_satisfaction_tracking(
            sentiment, escalation_risk, retrieved_articles
        )
        
        # Add metadata
        response.update({
            'sentiment_aware': True,
            'escalation_handled': escalation_analysis.get('escalation_needed', False),
            'articles_used': len(retrieved_articles),
            'confidence_score': sentiment_analysis.get('confidence', 0.0),
            'response_tone': self._determine_response_tone(sentiment, escalation_risk),
            'customer_id': customer_id,
            'timestamp': datetime.now().isoformat()
        })
        
        return response
    
    def _generate_ai_customer_response(self, 
                                     customer_query: str, 
                                     context_text: str, 
                                     sentiment_analysis: Dict[str, Any],
                                     escalation_analysis: Dict[str, Any],
                                     conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Generate AI-powered customer support response."""
        
        # Create system prompt for customer support
        system_prompt = self._create_customer_support_prompt(sentiment_analysis, escalation_analysis)
        
        # Build conversation context
        conversation_context = self._build_conversation_context(conversation_history)
        
        # Create user prompt
        user_prompt = f"""
Customer Query: {customer_query}

Relevant Help Articles:
{context_text}

Conversation History:
{conversation_context}

Please provide a helpful, empathetic response that:
1. Acknowledges the customer's sentiment appropriately
2. Provides clear, actionable solutions
3. Uses the right tone based on escalation risk
4. Includes relevant information from help articles
5. Offers next steps or follow-up options
"""
        
        try:
            import openai
            openai.api_key = self.openai_api_key
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            return {
                'content': ai_response,
                'response_type': 'ai_generated',
                'model_used': 'gpt-3.5-turbo'
            }
            
        except Exception as e:
            print(f"Error generating AI response: {e}")
            # Fallback to template response
            return self._generate_template_customer_response(
                customer_query, context_text, sentiment_analysis, escalation_analysis
            )
    
    def _generate_template_customer_response(self, 
                                           customer_query: str, 
                                           context_text: str, 
                                           sentiment_analysis: Dict[str, Any],
                                           escalation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate template-based customer support response."""
        
        sentiment = sentiment_analysis.get('overall_sentiment', 'neutral')
        escalation_risk = escalation_analysis.get('risk_level', 'low')
        
        # Comprehensive emotional intelligence system
        emotion_categories = {
            'joy_happiness': ['happy', 'happiness', 'joy', 'joyful', 'cheerful', 'delighted', 'pleased', 'thrilled', 'excited', 'elated', 'ecstatic', 'blissful', 'content', 'satisfied', 'grateful', 'thankful', 'blessed', 'wonderful', 'amazing', 'great', 'fantastic', 'awesome', 'brilliant', 'perfect', 'lovely', 'beautiful', 'sweet', 'nice'],
            
            'sadness_depression': ['sad', 'sadness', 'depressed', 'depression', 'melancholy', 'gloomy', 'down', 'blue', 'miserable', 'unhappy', 'disappointed', 'heartbroken', 'devastated', 'hopeless', 'despair', 'desperate', 'lonely', 'isolated', 'abandoned', 'rejected', 'worthless', 'useless', 'empty', 'numb', 'crying', 'tears', 'sobbing', 'weeping'],
            
            'anger_frustration': ['angry', 'anger', 'furious', 'rage', 'irritated', 'annoyed', 'frustrated', 'mad', 'livid', 'outraged', 'fuming', 'seething', 'hostile', 'aggressive', 'bitter', 'resentful', 'hate', 'hatred', 'disgusted', 'repulsed', 'offended', 'insulted', 'betrayed', 'betrayal'],
            
            'fear_anxiety': ['afraid', 'scared', 'fearful', 'terrified', 'panicked', 'anxious', 'anxiety', 'worried', 'concerned', 'nervous', 'tense', 'stressed', 'overwhelmed', 'overwhelming', 'paranoid', 'suspicious', 'doubtful', 'uncertain', 'confused', 'lost', 'helpless', 'powerless', 'vulnerable', 'exposed'],
            
            'surprise_shock': ['surprised', 'shocked', 'astonished', 'amazed', 'stunned', 'bewildered', 'confused', 'perplexed', 'puzzled', 'startled', 'alarmed', 'startled', 'taken aback', 'caught off guard', 'unexpected', 'unbelievable', 'incredible'],
            
            'disgust_contempt': ['disgusted', 'disgust', 'repulsed', 'revolted', 'sickened', 'nauseated', 'contempt', 'contemptuous', 'scornful', 'disdainful', 'disrespectful', 'offended', 'insulted', 'humiliated', 'embarrassed', 'ashamed', 'guilty', 'remorseful'],
            
            'love_affection': ['love', 'loving', 'affectionate', 'caring', 'tender', 'warm', 'fond', 'adore', 'adoration', 'passionate', 'romantic', 'intimate', 'close', 'connected', 'bonded', 'attached', 'devoted', 'loyal', 'faithful', 'trusting', 'trust'],
            
            'pride_confidence': ['proud', 'pride', 'confident', 'confidence', 'assured', 'certain', 'secure', 'strong', 'powerful', 'capable', 'competent', 'successful', 'accomplished', 'achieved', 'victorious', 'triumphant', 'empowered', 'motivated', 'inspired', 'determined'],
            
            'shame_embarrassment': ['ashamed', 'shame', 'embarrassed', 'embarrassment', 'humiliated', 'humiliation', 'mortified', 'guilty', 'guilt', 'remorseful', 'remorse', 'regretful', 'regret', 'sorry', 'apologetic', 'inferior', 'inadequate', 'worthless', 'useless'],
            
            'jealousy_envy': ['jealous', 'jealousy', 'envious', 'envy', 'covetous', 'resentful', 'bitter', 'spiteful', 'malicious', 'vengeful', 'vengeance', 'revenge', 'competitive', 'comparison', 'inferior', 'inadequate'],
            
            'hope_optimism': ['hopeful', 'hope', 'optimistic', 'optimism', 'positive', 'encouraged', 'inspired', 'motivated', 'determined', 'resilient', 'persevering', 'persistent', 'faithful', 'believing', 'trusting', 'confident', 'assured'],
            
            'gratitude_appreciation': ['grateful', 'gratitude', 'thankful', 'thanks', 'appreciative', 'appreciation', 'blessed', 'fortunate', 'lucky', 'privileged', 'honored', 'valued', 'cherished', 'treasured', 'precious', 'special'],
            
            'curiosity_interest': ['curious', 'curiosity', 'interested', 'interest', 'intrigued', 'fascinated', 'captivated', 'engaged', 'involved', 'absorbed', 'focused', 'attentive', 'alert', 'aware', 'mindful', 'present'],
            
            'boredom_apathy': ['bored', 'boredom', 'apathetic', 'apathy', 'indifferent', 'uninterested', 'disengaged', 'disconnected', 'distant', 'cold', 'numb', 'empty', 'hollow', 'meaningless', 'pointless', 'useless', 'unmotivated', 'lazy', 'tired', 'exhausted', 'drained', 'uninspired', 'stuck', 'trapped'],
            
            'excitement_enthusiasm': ['excited', 'excitement', 'enthusiastic', 'enthusiasm', 'eager', 'eagerness', 'thrilled', 'thrilling', 'energetic', 'energized', 'vibrant', 'lively', 'dynamic', 'passionate', 'zealous', 'fervent'],
            
            'calm_peace': ['calm', 'peaceful', 'peace', 'serene', 'tranquil', 'relaxed', 'relaxing', 'soothing', 'comfortable', 'comforted', 'safe', 'secure', 'protected', 'sheltered', 'nurtured', 'cared for', 'supported']
        }
        
        query_lower = customer_query.lower()
        
        # Detect which emotion category the query belongs to (with priority scoring)
        emotion_scores = {}
        for emotion_category, keywords in emotion_categories.items():
            score = 0
            for keyword in keywords:
                if keyword in query_lower:
                    score += 1
            if score > 0:
                emotion_scores[emotion_category] = score
        
        # Get the emotion with the highest score
        detected_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0] if emotion_scores else None
        
        # Debug: Print detected emotions for troubleshooting
        if emotion_scores:
            print(f"DEBUG: Detected emotions for '{customer_query}': {emotion_scores}")
            print(f"DEBUG: Selected emotion: {detected_emotion}")
        
        # Determine if it's a positive, negative, or neutral emotion
        positive_emotions = ['joy_happiness', 'love_affection', 'pride_confidence', 'hope_optimism', 'gratitude_appreciation', 'curiosity_interest', 'excitement_enthusiasm', 'calm_peace']
        negative_emotions = ['sadness_depression', 'anger_frustration', 'fear_anxiety', 'disgust_contempt', 'shame_embarrassment', 'jealousy_envy', 'boredom_apathy']
        
        is_positive_emotion = detected_emotion in positive_emotions
        is_emotional_query = detected_emotion is not None
        
        # Get appropriate empathy template based on detected emotion
        if detected_emotion and detected_emotion in self.empathy_templates:
            empathy_template = self.empathy_templates[detected_emotion]
        elif sentiment == 'negative' and escalation_analysis.get('risk_factors', {}).get('frustration_score', 0) > 0.7:
            empathy_template = self.empathy_templates['high_frustration']
        elif sentiment == 'negative' or escalation_analysis.get('risk_factors', {}).get('frustration_score', 0) > 0.3:
            empathy_template = self.empathy_templates['medium_frustration']
        else:
            empathy_template = self.empathy_templates['low_frustration']
        
        # Build response
        response_parts = []
        
        # Add empathetic acknowledgment
        acknowledgment = random.choice(empathy_template['acknowledgment'])
        response_parts.append(acknowledgment)
        
        # Add solution from articles
        if context_text and context_text != "No specific help articles found.":
            solution = self._extract_solution_from_articles(context_text)
            response_parts.append(f"\n{solution}")
        else:
            if detected_emotion:
                if detected_emotion in positive_emotions:
                    response_parts.append("\nI hope you continue to feel this way! Is there anything I can help you with today?")
                elif detected_emotion in negative_emotions:
                    response_parts.append("\nHere are some ways I can help you right now:")
                    response_parts.append("• I can connect you with our emotional support specialists who are trained to help")
                    response_parts.append("• We can work through this together step by step")
                    response_parts.append("• I'm here to listen and provide the support you need")
                else:
                    response_parts.append("\nI'm here to support you with whatever you need. How can I help?")
            else:
                response_parts.append("\nI'll help you with this right away. Let me provide you with the information you need.")
        
        # Add escalation response if needed
        if escalation_analysis.get('escalation_needed', False):
            escalation_response = self.escalation_responses[escalation_risk]['immediate']
            response_parts.append(f"\n{escalation_response}")
        
        # Add closing
        closing = self._generate_closing_phrase(sentiment, escalation_risk, detected_emotion)
        response_parts.append(f"\n{closing}")
        
        return {
            'content': ' '.join(response_parts),
            'response_type': 'template_generated',
            'tone_used': empathy_template['tone']
        }
    
    def _create_customer_support_prompt(self, sentiment_analysis: Dict[str, Any], escalation_analysis: Dict[str, Any]) -> str:
        """Create system prompt for customer support AI."""
        
        sentiment = sentiment_analysis.get('overall_sentiment', 'neutral')
        escalation_risk = escalation_analysis.get('risk_level', 'low')
        
        prompt = f"""You are an empathetic and effective customer support representative. 

Current Analysis:
- Customer Sentiment: {sentiment}
- Escalation Risk: {escalation_risk}
- Escalation Needed: {escalation_analysis.get('escalation_needed', False)}

Guidelines:
1. Always acknowledge the customer's feelings appropriately
2. Use empathetic language that matches the sentiment level
3. Provide clear, actionable solutions
4. If escalation is needed, explain the process clearly
5. Be professional but warm and understanding
6. Offer specific next steps or follow-up options
7. Use the customer's name if available
8. Keep responses concise but comprehensive

Tone Guidelines:
- For negative sentiment: Be extra empathetic and understanding
- For high escalation risk: Be urgent and proactive
- For positive sentiment: Be enthusiastic and helpful
- For neutral sentiment: Be professional and friendly

Remember: The goal is to resolve the customer's issue while making them feel heard and valued."""
        
        return prompt
    
    def _format_article_context(self, retrieved_articles: List[Dict[str, Any]]) -> str:
        """Format retrieved articles into context for response generation."""
        if not retrieved_articles:
            return "No specific help articles found."
        
        context_parts = []
        for i, article in enumerate(retrieved_articles[:3], 1):  # Limit to top 3 articles
            # Handle different article formats
            title = article.get('title', article.get('metadata', {}).get('title', 'No title'))
            content = article.get('content', 'No content')
            category = article.get('category', article.get('metadata', {}).get('category', 'Unknown'))
            
            context_parts.append(f"Article {i}: {title}")
            context_parts.append(f"Content: {content[:200]}...")
            context_parts.append(f"Category: {category}")
            context_parts.append("---")
        
        return "\n".join(context_parts)
    
    def _extract_solution_from_articles(self, context_text: str) -> str:
        """Extract solution steps from article context."""
        # Simple extraction - look for numbered steps
        lines = context_text.split('\n')
        solution_lines = []
        
        for line in lines:
            if re.match(r'^\d+\.', line.strip()):  # Numbered steps
                solution_lines.append(line.strip())
            elif 'step' in line.lower() and len(line.strip()) > 20:
                solution_lines.append(line.strip())
        
        if solution_lines:
            return "Here's what you can do:\n" + "\n".join(solution_lines[:5])  # Limit to 5 steps
        else:
            return "Based on our help articles, I can provide you with the information you need."
    
    def _add_escalation_handling(self, escalation_risk: str, escalation_analysis: Dict[str, Any]) -> str:
        """Add escalation handling to response."""
        escalation_response = self.escalation_responses[escalation_risk]['immediate']
        
        if escalation_analysis.get('recommendations'):
            recommendations = escalation_analysis['recommendations'][:2]  # Limit to 2 recommendations
            escalation_response += "\n\n" + "\n".join(recommendations)
        
        return escalation_response
    
    def _generate_satisfaction_tracking(self, sentiment: str, escalation_risk: str, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate satisfaction tracking information."""
        return {
            'expected_satisfaction': self._predict_satisfaction(sentiment, escalation_risk, articles),
            'follow_up_needed': escalation_risk in ['high', 'critical'],
            'follow_up_timing': self._get_follow_up_timing(escalation_risk),
            'satisfaction_phrases': self.satisfaction_phrases.get(sentiment, self.satisfaction_phrases['neutral'])
        }
    
    def _predict_satisfaction(self, sentiment: str, escalation_risk: str, articles: List[Dict[str, Any]]) -> float:
        """Predict customer satisfaction based on various factors."""
        base_satisfaction = {
            'positive': 0.8,
            'neutral': 0.6,
            'negative': 0.4
        }.get(sentiment, 0.6)
        
        # Adjust based on escalation risk
        risk_adjustment = {
            'critical': -0.3,
            'high': -0.2,
            'medium': -0.1,
            'low': 0.0,
            'minimal': 0.1
        }.get(escalation_risk, 0.0)
        
        # Adjust based on article quality
        if articles:
            avg_helpful_rate = sum(article.get('satisfaction_metrics', {}).get('helpful_rate', 0.5) 
                                 for article in articles) / len(articles)
            article_adjustment = (avg_helpful_rate - 0.5) * 0.2
        else:
            article_adjustment = -0.1
        
        satisfaction = base_satisfaction + risk_adjustment + article_adjustment
        return max(0.0, min(1.0, satisfaction))
    
    def _get_follow_up_timing(self, escalation_risk: str) -> str:
        """Get recommended follow-up timing."""
        timing_map = {
            'critical': 'within 30 minutes',
            'high': 'within 2 hours',
            'medium': 'within 24 hours',
            'low': 'within 48 hours',
            'minimal': 'within 1 week'
        }
        return timing_map.get(escalation_risk, 'within 24 hours')
    
    def _determine_response_tone(self, sentiment: str, escalation_risk: str) -> str:
        """Determine the appropriate response tone."""
        if escalation_risk == 'critical':
            return 'urgent and empathetic'
        elif escalation_risk == 'high':
            return 'attentive and understanding'
        elif sentiment == 'negative':
            return 'empathetic and supportive'
        elif sentiment == 'positive':
            return 'enthusiastic and helpful'
        else:
            return 'professional and friendly'
    
    def _generate_closing_phrase(self, sentiment: str, escalation_risk: str, detected_emotion: str = None) -> str:
        """Generate appropriate closing phrase."""
        if detected_emotion:
            emotion_closings = {
                'joy_happiness': "Keep spreading that joy! I'm here if you need anything else.",
                'sadness_depression': "Remember, you're not alone in this. I'm here to support you, and we have specialists ready to help if you need more assistance. Take care of yourself.",
                'anger_frustration': "I'm here to help you work through this. Don't hesitate to reach out if you need more support.",
                'fear_anxiety': "You're safe here, and I'm here to help you feel more secure. Take care of yourself.",
                'surprise_shock': "Take your time to process this. I'm here if you need to talk more about it.",
                'disgust_contempt': "I understand your feelings, and I'm here to help you address this situation.",
                'love_affection': "Your love and care are beautiful. I'm here to support you in any way I can.",
                'pride_confidence': "Your confidence is inspiring! Keep up the great work, and I'm here if you need anything.",
                'shame_embarrassment': "Everyone makes mistakes, and you're still valued. I'm here to support you.",
                'jealousy_envy': "These feelings are natural, and I'm here to help you work through them in a healthy way.",
                'hope_optimism': "Your optimism is wonderful! Keep that positive energy, and I'm here to support you.",
                'gratitude_appreciation': "Your gratitude is beautiful. I'm grateful to be here for you.",
                'curiosity_interest': "Your curiosity is wonderful! I'm here to help you explore and learn more.",
                'boredom_apathy': "I'm here to help you find something engaging and meaningful. Let's work on this together.",
                'excitement_enthusiasm': "Your excitement is contagious! Keep that energy, and I'm here to support you.",
                'calm_peace': "Your peace is beautiful. I'm here to help you maintain that serenity."
            }
            return emotion_closings.get(detected_emotion, "I'm here to support you with whatever you need. Take care!")
        elif escalation_risk in ['critical', 'high']:
            return "I'm committed to resolving this for you quickly. Please don't hesitate to reach out if you need immediate assistance."
        elif sentiment == 'negative':
            return "I want to make sure we find the right solution for you. Please let me know if there's anything else I can do to help."
        elif sentiment == 'positive':
            return "I'm glad I could help! Please let me know if you need anything else."
        else:
            return "I'm here if you have any other questions. Thank you for reaching out to us."
    
    def _build_conversation_context(self, conversation_history: List[Dict] = None) -> str:
        """Build conversation context from history."""
        if not conversation_history:
            return "No previous conversation history."
        
        context_parts = []
        for i, conv in enumerate(conversation_history[-3:], 1):  # Last 3 conversations
            context_parts.append(f"Previous {i}: {conv.get('query', 'No query')}")
            context_parts.append(f"Response: {conv.get('response', 'No response')[:100]}...")
            context_parts.append("---")
        
        return "\n".join(context_parts)
    
    def generate_follow_up_response(self, 
                                  customer_id: str,
                                  previous_response: Dict[str, Any],
                                  satisfaction_score: float = None) -> str:
        """Generate follow-up response based on customer satisfaction."""
        
        if satisfaction_score is None:
            return "I wanted to follow up on your recent inquiry. How are things going? Is there anything else I can help you with?"
        
        if satisfaction_score >= 0.8:
            return "I'm glad I could help resolve your issue! Is there anything else you'd like assistance with?"
        elif satisfaction_score >= 0.6:
            return "I hope the solution I provided was helpful. If you need any clarification or have other questions, please let me know."
        else:
            return "I want to make sure we've fully resolved your issue. Could you let me know if there's anything else I can do to help?"
    
    def validate_response_quality(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the quality of the generated response."""
        content = response.get('content', '')
        
        quality_metrics = {
            'length_appropriate': 50 <= len(content) <= 500,
            'has_acknowledgment': any(word in content.lower() for word in ['understand', 'hear', 'see']),
            'has_solution': any(word in content.lower() for word in ['can', 'will', 'here', 'solution']),
            'has_closing': any(word in content.lower() for word in ['thank', 'help', 'assist', 'reach']),
            'tone_appropriate': True  # Would need more sophisticated analysis
        }
        
        quality_score = sum(quality_metrics.values()) / len(quality_metrics)
        
        return {
            'quality_score': quality_score,
            'metrics': quality_metrics,
            'recommendations': self._get_quality_recommendations(quality_metrics)
        }
    
    def _get_quality_recommendations(self, metrics: Dict[str, bool]) -> List[str]:
        """Get recommendations for improving response quality."""
        recommendations = []
        
        if not metrics['length_appropriate']:
            recommendations.append("Consider adjusting response length for better engagement")
        
        if not metrics['has_acknowledgment']:
            recommendations.append("Include empathetic acknowledgment of customer's situation")
        
        if not metrics['has_solution']:
            recommendations.append("Provide clear, actionable solutions")
        
        if not metrics['has_closing']:
            recommendations.append("Include appropriate closing phrase")
        
        return recommendations 
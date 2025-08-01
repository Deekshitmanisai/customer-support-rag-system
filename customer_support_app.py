#!/usr/bin/env python3
"""
Customer Support RAG System - Web Application
============================================

A comprehensive web application for the Customer Support RAG System with:
- Real-time customer query processing
- Sentiment analysis and escalation prediction
- Empathetic response generation
- Customer satisfaction tracking
- System analytics and reporting
"""

import streamlit as st
import time
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

# Import our Customer Support RAG System
from src.customer_support_rag_system import CustomerSupportRAGSystem

class CustomerSupportApp:
    """Customer Support RAG System Web Application."""
    
    def __init__(self):
        """Initialize the customer support application."""
        st.set_page_config(
            page_title="Customer Support RAG System",
            page_icon="🎧",
            layout="wide",
            initial_sidebar_state="expanded"
                )
        
        # Add custom CSS for modern dark sidebar
        st.markdown("""
        <style>
        .css-1d391kg {
            background-color: #2c3e50 !important;
        }
        [data-testid="stSidebar"] {
            background-color: #2c3e50 !important;
            padding: 1.5rem !important;
            border-right: 3px solid #34495e !important;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #ecf0f1 !important;
            font-weight: 700 !important;
            margin-bottom: 1.2rem !important;
            font-size: 1.2rem !important;
            border-bottom: 2px solid #3498db !important;
            padding-bottom: 0.8rem !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
        }
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] div, [data-testid="stSidebar"] span {
            color: #bdc3c7 !important;
            font-weight: 500 !important;
        }
        [data-testid="stSidebar"] .stButton > button {
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            padding: 0.8rem 1.5rem !important;
            margin: 0.8rem 0 !important;
            box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3) !important;
            transition: all 0.3s ease !important;
            font-size: 1rem !important;
            width: 100% !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: linear-gradient(135deg, #2980b9 0%, #1f5f8b 100%) !important;
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 25px rgba(52, 152, 219, 0.4) !important;
        }
        [data-testid="stSidebar"] .stSelectbox > div > div {
            background-color: #34495e !important;
            color: #ecf0f1 !important;
            border: 2px solid #3498db !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 0.8rem !important;
            margin: 0.8rem 0 !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
            font-size: 1rem !important;
        }
        [data-testid="stSidebar"] .stText {
            color: #bdc3c7 !important;
            font-weight: 500 !important;
            margin: 0.8rem 0 !important;
            font-size: 1rem !important;
        }
        [data-testid="stSidebar"] .stMetric {
            background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%) !important;
            border-radius: 12px !important;
            padding: 1rem !important;
            margin: 0.8rem 0 !important;
            border: 2px solid #3498db !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
        }
        [data-testid="stSidebar"] .stSuccess {
            background: linear-gradient(135deg, #27ae60 0%, #229954 100%) !important;
            color: white !important;
            border-radius: 10px !important;
            padding: 0.8rem !important;
            margin: 0.8rem 0 !important;
            border: 2px solid #2ecc71 !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        
        # Initialize the Customer Support RAG System
        if 'customer_support_system' not in st.session_state:
            with st.spinner("🚀 Initializing Customer Support RAG System..."):
                st.session_state.customer_support_system = CustomerSupportRAGSystem()
        
        # Initialize session state
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'current_customer_id' not in st.session_state:
            st.session_state.current_customer_id = None
        if 'satisfaction_feedback' not in st.session_state:
            st.session_state.satisfaction_feedback = {}
    
    def run(self):
        """Run the customer support application."""
        # Beautiful header with gradient background
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        ">
            <h1 style="
                color: white;
                font-size: 3rem;
                font-weight: 700;
                margin: 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            ">🎧 Customer Support RAG System</h1>
            <p style="
                color: rgba(255,255,255,0.9);
                font-size: 1.2rem;
                margin: 0.5rem 0 0 0;
                font-weight: 300;
            ">Intelligent, Empathetic, and Effective Customer Support</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar navigation
        self._create_sidebar()
        
        # Main content area
        page = st.sidebar.selectbox(
            "Navigation",
            ["Customer Support", "Analytics Dashboard", "System Testing", "Reports & Export"]
        )
        
        if page == "Customer Support":
            self.customer_support_page()
        elif page == "Analytics Dashboard":
            self.analytics_dashboard_page()
        elif page == "System Testing":
            self.system_testing_page()
        elif page == "Reports & Export":
            self.reports_export_page()
    
    def _create_sidebar(self):
        """Create the sidebar with system information and controls."""
        st.sidebar.title("🎧 Customer Support")
        
        # System status
        st.sidebar.subheader("System Status")
        
        # Test system functionality
        if st.sidebar.button("🔍 Test System"):
            with st.spinner("Testing system components..."):
                test_results = st.session_state.customer_support_system.test_system_functionality()
                
                # Display test results
                for component, result in test_results.items():
                    if component != 'overall_system_health':
                        status = "✅" if result.get('status') == 'passed' else "❌"
                        st.sidebar.text(f"{status} {component}")
                
                # Overall health
                overall = test_results.get('overall_system_health', {})
                health_score = overall.get('score', 0)
                health_status = overall.get('status', 'unknown')
                
                st.sidebar.metric("System Health", f"{health_score:.1%}", health_status)
        
        # Customer ID management
        st.sidebar.subheader("Customer Management")
        
        if st.sidebar.button("🆔 New Customer"):
            st.session_state.current_customer_id = None
            st.session_state.conversation_history = []
            st.sidebar.success("New customer session started!")
        
        if st.session_state.current_customer_id:
            st.sidebar.text(f"Current Customer: {st.session_state.current_customer_id[:8]}...")
        
        # System controls
        st.sidebar.subheader("System Controls")
        
        if st.sidebar.button("📊 Refresh Analytics"):
            st.rerun()
        
        if st.sidebar.button("🗑️ Clear History"):
            st.session_state.conversation_history = []
            st.sidebar.success("Conversation history cleared!")
    
    def customer_support_page(self):
        """Main customer support page."""
        
        # Create two columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._customer_chat_interface()
        
        with col2:
            self._customer_insights_panel()
    
    def _customer_chat_interface(self):
        """Customer chat interface."""
        st.subheader("💬 Customer Chat")
        
        # Customer query input
        customer_query = st.text_area(
            "Customer Query:",
            placeholder="Enter the customer's question or issue...",
            height=100
        )
        
        # Process query button
        if st.button("🚀 Process Query", type="primary"):
            if customer_query.strip():
                self._process_customer_query(customer_query)
            else:
                st.warning("Please enter a customer query.")
        
        # Display conversation history
        if st.session_state.conversation_history:
            st.subheader("📝 Conversation History")
            
            for i, interaction in enumerate(st.session_state.conversation_history):
                with st.expander(f"Interaction {i+1} - {interaction.get('timestamp', 'Unknown')}"):
                    self._display_interaction(interaction)
    
    def _process_customer_query(self, query: str):
        """Process a customer query."""
        with st.spinner("Processing customer query..."):
            # Process the query
            result = st.session_state.customer_support_system.process_customer_query(
                customer_query=query,
                customer_id=st.session_state.current_customer_id,
                conversation_history=st.session_state.conversation_history
            )
            
            # Update current customer ID
            if not st.session_state.current_customer_id:
                st.session_state.current_customer_id = result['customer_id']
            
            # Add to conversation history
            st.session_state.conversation_history.append(result)
            
            # Display results
            st.success("✅ Query processed successfully!")
            
            # Show response
            self._display_response(result)
    
    def _display_response(self, result: dict):
        """Display the system response."""
        response = result.get('response', {})
        
        # Response content
        st.subheader("🤖 System Response")
        st.write(response.get('content', 'No response generated'))
        
        # Response metadata
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Response Time", f"{result.get('response_time', 0):.2f}s")
        
        with col2:
            st.metric("Articles Used", result.get('system_metadata', {}).get('articles_used', 0))
        
        with col3:
            st.metric("Confidence", f"{result.get('system_metadata', {}).get('confidence_score', 0):.1%}")
        
        # Detailed analysis
        with st.expander("🔍 Detailed Analysis"):
            self._display_detailed_analysis(result)
    
    def _display_detailed_analysis(self, result: dict):
        """Display detailed analysis of the response."""
        # Sentiment analysis
        sentiment = result.get('sentiment_analysis', {})
        st.subheader("😊 Sentiment Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Sentiment", sentiment.get('overall_sentiment', 'Unknown'))
        with col2:
            st.metric("Confidence", f"{sentiment.get('confidence', 0):.1%}")
        with col3:
            st.metric("Polarity", f"{sentiment.get('polarity', 0):.2f}")
        
        # Categorization
        categorization = result.get('categorization', {})
        st.subheader("📂 Query Categorization")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Category", categorization.get('category', 'Unknown'))
        with col2:
            st.metric("Confidence", f"{categorization.get('confidence', 0):.1%}")
        
        # Escalation analysis
        escalation = result.get('escalation_analysis', {})
        st.subheader("⚠️ Escalation Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk Level", escalation.get('risk_level', 'Unknown'))
        with col2:
            st.metric("Escalation Needed", "Yes" if escalation.get('escalation_needed') else "No")
        with col3:
            st.metric("Confidence", f"{escalation.get('confidence', 0):.1%}")
        
        # Risk factors
        risk_factors = escalation.get('risk_factors', {})
        if risk_factors:
            st.subheader("📊 Risk Factors")
            risk_df = pd.DataFrame(list(risk_factors.items()), columns=['Factor', 'Score'])
            st.bar_chart(risk_df.set_index('Factor'))
        
        # Recommendations
        recommendations = escalation.get('recommendations', [])
        if recommendations:
            st.subheader("💡 Recommendations")
            for rec in recommendations:
                st.write(f"• {rec}")
    
    def _display_interaction(self, interaction: dict):
        """Display a single interaction."""
        st.write(f"**Query:** {interaction.get('query', 'No query')}")
        st.write(f"**Response:** {interaction.get('response', {}).get('content', 'No response')}")
        
        # Quick metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Time", f"{interaction.get('response_time', 0):.2f}s")
        with col2:
            st.metric("Sentiment", interaction.get('sentiment_analysis', {}).get('overall_sentiment', 'Unknown'))
        with col3:
            escalation_needed = interaction.get('escalation_analysis', {}).get('escalation_needed', False)
            st.metric("Escalation", "Yes" if escalation_needed else "No")
    
    def _customer_insights_panel(self):
        """Customer insights panel."""
        st.subheader("👤 Customer Insights")
        
        if st.session_state.current_customer_id:
            # Get customer insights
            insights = st.session_state.customer_support_system.get_customer_insights(
                st.session_state.current_customer_id
            )
            
            if 'error' not in insights:
                # Customer metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Interactions", insights.get('total_interactions', 0))
                with col2:
                    st.metric("Avg Satisfaction", f"{insights.get('average_satisfaction', 0):.1%}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Escalation Rate", f"{insights.get('escalation_rate', 0):.1%}")
                with col2:
                    st.metric("Risk Level", insights.get('risk_level', 'Unknown'))
                
                # Recommendations
                recommendations = insights.get('recommendations', [])
                if recommendations:
                    st.subheader("💡 Recommendations")
                    for rec in recommendations:
                        st.write(f"• {rec}")
            else:
                st.info("No customer insights available yet.")
        else:
            st.info("Start a customer interaction to see insights.")
        
        # Satisfaction feedback
        st.subheader("⭐ Satisfaction Feedback")
        
        if st.session_state.conversation_history:
            latest_interaction = st.session_state.conversation_history[-1]
            
            satisfaction_score = st.slider(
                "Rate your satisfaction (1-5):",
                min_value=1,
                max_value=5,
                value=3,
                key="satisfaction_slider"
            )
            
            feedback = st.text_area(
                "Additional feedback (optional):",
                placeholder="Share your thoughts about the response...",
                key="feedback_text"
            )
            
            if st.button("📝 Submit Feedback"):
                if st.session_state.current_customer_id:
                    # Convert 1-5 scale to 0-1 scale
                    normalized_score = (satisfaction_score - 1) / 4
                    
                    result = st.session_state.customer_support_system.record_satisfaction_feedback(
                        customer_id=st.session_state.current_customer_id,
                        satisfaction_score=normalized_score,
                        feedback=feedback if feedback.strip() else None
                    )
                    
                    if 'error' not in result:
                        st.success("✅ Feedback recorded successfully!")
                    else:
                        st.error(f"❌ Error recording feedback: {result['error']}")
                else:
                    st.warning("No active customer session.")
    
    def analytics_dashboard_page(self):
        """Analytics dashboard page."""
        st.header("📊 Analytics Dashboard")
        
        # Time period selector
        time_period = st.selectbox(
            "Select Time Period:",
            ["all", "today", "week", "month"],
            format_func=lambda x: x.title()
        )
        
        # Get analytics
        analytics = st.session_state.customer_support_system.get_system_analytics(time_period)
        
        # Key metrics
        st.subheader("📈 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_interactions = analytics['satisfaction_analytics'].get('total_interactions', 0)
            st.metric("Total Interactions", total_interactions)
        
        with col2:
            unique_customers = analytics['satisfaction_analytics'].get('unique_customers', 0)
            st.metric("Unique Customers", unique_customers)
        
        with col3:
            avg_satisfaction = analytics['satisfaction_analytics'].get('satisfaction_distribution', {}).get('average_satisfaction', 0)
            st.metric("Avg Satisfaction", f"{avg_satisfaction:.1%}")
        
        with col4:
            escalation_rate = analytics['escalation_analytics'].get('escalation_rate', 0)
            st.metric("Escalation Rate", f"{escalation_rate:.1%}")
        
        # Create visualizations
        st.subheader("📊 Visualizations")
        
        try:
            visualizations = st.session_state.customer_support_system.create_system_visualizations(time_period)
            
            # Satisfaction charts
            if 'satisfaction_charts' in visualizations:
                st.plotly_chart(visualizations['satisfaction_charts'], use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating visualizations: {e}")
        
        # Detailed analytics
        with st.expander("🔍 Detailed Analytics"):
            self._display_detailed_analytics(analytics)
    
    def _display_detailed_analytics(self, analytics: dict):
        """Display detailed analytics."""
        # Satisfaction distribution
        satisfaction_dist = analytics['satisfaction_analytics'].get('satisfaction_distribution', {})
        if 'distribution' in satisfaction_dist:
            st.subheader("😊 Satisfaction Distribution")
            dist_data = satisfaction_dist['distribution']
            
            if dist_data:
                dist_df = pd.DataFrame([
                    {'Level': level, 'Count': data['count'], 'Percentage': data['percentage']}
                    for level, data in dist_data.items()
                ])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.bar_chart(dist_df.set_index('Level')['Count'])
                with col2:
                    st.bar_chart(dist_df.set_index('Level')['Percentage'])
        
        # Response type performance
        response_performance = analytics['satisfaction_analytics'].get('response_type_performance', {})
        if response_performance:
            st.subheader("🤖 Response Type Performance")
            
            perf_data = []
            for response_type, perf in response_performance.items():
                perf_data.append({
                    'Response Type': response_type,
                    'Count': perf['count'],
                    'Avg Satisfaction': perf['average_satisfaction']
                })
            
            if perf_data:
                perf_df = pd.DataFrame(perf_data)
                st.bar_chart(perf_df.set_index('Response Type')['Avg Satisfaction'])
        
        # Escalation analysis
        escalation_analysis = analytics['satisfaction_analytics'].get('escalation_analysis', {})
        if escalation_analysis:
            st.subheader("⚠️ Escalation Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Escalations", escalation_analysis.get('total_escalations', 0))
            with col2:
                st.metric("Escalated Satisfaction", f"{escalation_analysis.get('escalated_satisfaction', 0):.1%}")
            with col3:
                st.metric("Non-Escalated Satisfaction", f"{escalation_analysis.get('non_escalated_satisfaction', 0):.1%}")
    
    def system_testing_page(self):
        """System testing page."""
        st.header("🧪 System Testing")
        
        # Test all components
        if st.button("🔍 Run Full System Test", type="primary"):
            with st.spinner("Running comprehensive system tests..."):
                test_results = st.session_state.customer_support_system.test_system_functionality()
                
                # Display test results
                st.subheader("📋 Test Results")
                
                for component, result in test_results.items():
                    if component != 'overall_system_health':
                        status = "✅" if result.get('status') == 'passed' else "❌"
                        st.write(f"{status} **{component.replace('_', ' ').title()}**: {result.get('message', 'No message')}")
                
                # Overall health
                overall = test_results.get('overall_system_health', {})
                health_score = overall.get('score', 0)
                health_status = overall.get('status', 'unknown')
                
                st.subheader("🏥 System Health")
                st.metric("Overall Health", f"{health_score:.1%}", health_status)
                
                # Health indicator
                if health_score >= 0.8:
                    st.success("🎉 System is healthy and ready for production!")
                elif health_score >= 0.6:
                    st.warning("⚠️ System needs attention but is functional.")
                else:
                    st.error("🚨 System has critical issues that need immediate attention.")
        
        # Individual component tests
        st.subheader("🔧 Individual Component Tests")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🤖 Test RAG System"):
                with st.spinner("Testing RAG system..."):
                    result = st.session_state.customer_support_system._test_rag_system()
                    status = "✅" if result.get('status') == 'passed' else "❌"
                    st.write(f"{status} {result.get('message', 'No message')}")
            
            if st.button("📚 Test Knowledge Base"):
                with st.spinner("Testing knowledge base..."):
                    result = st.session_state.customer_support_system._test_knowledge_base()
                    status = "✅" if result.get('status') == 'passed' else "❌"
                    st.write(f"{status} {result.get('message', 'No message')}")
            
            if st.button("⚠️ Test Escalation Predictor"):
                with st.spinner("Testing escalation predictor..."):
                    result = st.session_state.customer_support_system._test_escalation_predictor()
                    status = "✅" if result.get('status') == 'passed' else "❌"
                    st.write(f"{status} {result.get('message', 'No message')}")
        
        with col2:
            if st.button("💬 Test Response Generator"):
                with st.spinner("Testing response generator..."):
                    result = st.session_state.customer_support_system._test_response_generator()
                    status = "✅" if result.get('status') == 'passed' else "❌"
                    st.write(f"{status} {result.get('message', 'No message')}")
            
            if st.button("📊 Test Satisfaction Tracker"):
                with st.spinner("Testing satisfaction tracker..."):
                    result = st.session_state.customer_support_system._test_satisfaction_tracker()
                    status = "✅" if result.get('status') == 'passed' else "❌"
                    st.write(f"{status} {result.get('message', 'No message')}")
            
            if st.button("😊 Test Sentiment Analyzer"):
                with st.spinner("Testing sentiment analyzer..."):
                    result = st.session_state.customer_support_system._test_sentiment_analyzer()
                    status = "✅" if result.get('status') == 'passed' else "❌"
                    st.write(f"{status} {result.get('message', 'No message')}")
    
    def reports_export_page(self):
        """Reports and export page."""
        st.header("📋 Reports & Export")
        
        # Generate reports
        st.subheader("📊 Generate Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            time_period = st.selectbox(
                "Select Time Period:",
                ["all", "today", "week", "month"],
                format_func=lambda x: x.title(),
                key="report_time_period"
            )
            
            if st.button("📋 Generate Comprehensive Report", type="primary"):
                with st.spinner("Generating comprehensive report..."):
                    report = st.session_state.customer_support_system.generate_comprehensive_report(time_period)
                    
                    st.subheader("📋 Comprehensive Report")
                    st.markdown(report)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Report (Markdown)",
                        data=report,
                        file_name=f"customer_support_report_{time_period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
        
        with col2:
            if st.button("📊 Generate Satisfaction Report"):
                with st.spinner("Generating satisfaction report..."):
                    report = st.session_state.customer_support_system.satisfaction_tracker.generate_satisfaction_report(time_period)
                    
                    st.subheader("😊 Satisfaction Report")
                    st.markdown(report)
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Satisfaction Report",
                        data=report,
                        file_name=f"satisfaction_report_{time_period}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
        
        # Export data
        st.subheader("📤 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Export System Data"):
                with st.spinner("Exporting system data..."):
                    filename = st.session_state.customer_support_system.export_system_data()
                    
                    # Read the exported file
                    with open(filename, 'r') as f:
                        data = f.read()
                    
                    st.success(f"✅ Data exported to {filename}")
                    
                    # Download button
                    st.download_button(
                        label="📥 Download System Data (JSON)",
                        data=data,
                        file_name=filename,
                        mime="application/json"
                    )
        
        with col2:
            if st.button("📊 Export Satisfaction Data"):
                with st.spinner("Exporting satisfaction data..."):
                    filename = st.session_state.customer_support_system.satisfaction_tracker.export_satisfaction_data()
                    
                    # Read the exported file
                    with open(filename, 'r') as f:
                        data = f.read()
                    
                    st.success(f"✅ Satisfaction data exported to {filename}")
                    
                    # Download button
                    st.download_button(
                        label="📥 Download Satisfaction Data (JSON)",
                        data=data,
                        file_name=filename,
                        mime="application/json"
                    )

def main():
    """Main function to run the customer support application."""
    app = CustomerSupportApp()
    app.run()

if __name__ == "__main__":
    main() 
"""
Advanced Analytics Dashboard for RAG Chatbot
Provides comprehensive usage metrics, performance tracking, and insights
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import streamlit as st
import pandas as pd

# Optional plotly imports for visualizations
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = go = make_subplots = None


@dataclass
class AnalyticsEvent:
    """Represents a single analytics event"""
    id: str
    timestamp: datetime
    event_type: str  # 'message', 'document_upload', 'query', 'rag_retrieval', 'model_switch'
    user_id: str = "anonymous"
    session_id: str = ""
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AnalyticsTracker:
    """Tracks and analyzes chatbot usage patterns"""
    
    def __init__(self, db_path: str = "analytics.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the analytics database"""
        with sqlite3.connect(self.db_path) as conn:
            # Events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analytics_events (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    user_id TEXT DEFAULT 'anonymous',
                    session_id TEXT DEFAULT '',
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Performance metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Document engagement table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_engagement (
                    id TEXT PRIMARY KEY,
                    document_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    engagement_type TEXT NOT NULL,
                    relevance_score REAL DEFAULT 0,
                    chunk_index INTEGER DEFAULT 0,
                    query TEXT DEFAULT ''
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON analytics_events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON analytics_events(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_engagement_document ON document_engagement(document_name)")
    
    def track_event(self, event: AnalyticsEvent) -> bool:
        """Track a single analytics event"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO analytics_events 
                    (id, timestamp, event_type, user_id, session_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    event.id,
                    event.timestamp.isoformat(),
                    event.event_type,
                    event.user_id,
                    event.session_id,
                    json.dumps(event.metadata)
                ))
            return True
        except Exception as e:
            st.error(f"Failed to track event: {e}")
            return False
    
    def track_performance(self, metric_type: str, value: float, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Track a performance metric"""
        try:
            if metadata is None:
                metadata = {}
            
            import hashlib
            metric_id = hashlib.md5(f"{metric_type}_{datetime.now().timestamp()}".encode()).hexdigest()[:12]
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO performance_metrics 
                    (id, timestamp, metric_type, value, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    metric_id,
                    datetime.now().isoformat(),
                    metric_type,
                    value,
                    json.dumps(metadata)
                ))
            return True
        except Exception:
            return False
    
    def track_document_engagement(
        self, 
        document_name: str, 
        engagement_type: str,
        relevance_score: float = 0,
        chunk_index: int = 0,
        query: str = ""
    ) -> bool:
        """Track document engagement events"""
        try:
            import hashlib
            engagement_id = hashlib.md5(f"{document_name}_{datetime.now().timestamp()}".encode()).hexdigest()[:12]
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO document_engagement 
                    (id, document_name, timestamp, engagement_type, relevance_score, chunk_index, query)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    engagement_id,
                    document_name,
                    datetime.now().isoformat(),
                    engagement_type,
                    relevance_score,
                    chunk_index,
                    query
                ))
            return True
        except Exception:
            return False
    
    def get_usage_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get usage statistics for the specified period"""
        try:
            since_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                # Total events
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM analytics_events WHERE timestamp >= ?", 
                    (since_date,)
                )
                total_events = cursor.fetchone()[0]
                
                # Events by type
                cursor = conn.execute("""
                    SELECT event_type, COUNT(*) as count 
                    FROM analytics_events 
                    WHERE timestamp >= ? 
                    GROUP BY event_type
                    ORDER BY count DESC
                """, (since_date,))
                events_by_type = dict(cursor.fetchall())
                
                # Daily activity
                cursor = conn.execute("""
                    SELECT DATE(timestamp) as date, COUNT(*) as count
                    FROM analytics_events 
                    WHERE timestamp >= ?
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                """, (since_date,))
                daily_activity = dict(cursor.fetchall())
                
                # Unique sessions
                cursor = conn.execute("""
                    SELECT COUNT(DISTINCT session_id) 
                    FROM analytics_events 
                    WHERE timestamp >= ? AND session_id != ''
                """, (since_date,))
                unique_sessions = cursor.fetchone()[0]
                
                return {
                    "total_events": total_events,
                    "events_by_type": events_by_type,
                    "daily_activity": daily_activity,
                    "unique_sessions": unique_sessions,
                    "period_days": days
                }
        except Exception:
            return {
                "total_events": 0,
                "events_by_type": {},
                "daily_activity": {},
                "unique_sessions": 0,
                "period_days": days
            }
    
    def get_document_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get document engagement analytics"""
        try:
            since_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                # Most engaged documents
                cursor = conn.execute("""
                    SELECT document_name, COUNT(*) as engagement_count,
                           AVG(relevance_score) as avg_relevance
                    FROM document_engagement 
                    WHERE timestamp >= ?
                    GROUP BY document_name
                    ORDER BY engagement_count DESC
                    LIMIT 10
                """, (since_date,))
                top_documents = [
                    {
                        "document": row[0],
                        "engagement_count": row[1],
                        "avg_relevance": round(row[2], 3) if row[2] else 0
                    }
                    for row in cursor.fetchall()
                ]
                
                # Engagement types
                cursor = conn.execute("""
                    SELECT engagement_type, COUNT(*) as count
                    FROM document_engagement 
                    WHERE timestamp >= ?
                    GROUP BY engagement_type
                """, (since_date,))
                engagement_types = dict(cursor.fetchall())
                
                # Average relevance over time
                cursor = conn.execute("""
                    SELECT DATE(timestamp) as date, AVG(relevance_score) as avg_relevance
                    FROM document_engagement 
                    WHERE timestamp >= ? AND relevance_score > 0
                    GROUP BY DATE(timestamp)
                    ORDER BY date
                """, (since_date,))
                relevance_over_time = dict(cursor.fetchall())
                
                return {
                    "top_documents": top_documents,
                    "engagement_types": engagement_types,
                    "relevance_over_time": relevance_over_time
                }
        except Exception:
            return {
                "top_documents": [],
                "engagement_types": {},
                "relevance_over_time": {}
            }
    
    def get_performance_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get performance analytics"""
        try:
            since_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                # Average response times
                cursor = conn.execute("""
                    SELECT metric_type, AVG(value) as avg_value, COUNT(*) as count
                    FROM performance_metrics 
                    WHERE timestamp >= ? AND metric_type LIKE '%_time'
                    GROUP BY metric_type
                """, (since_date,))
                avg_times = {
                    row[0]: {"avg": round(row[1], 3), "count": row[2]}
                    for row in cursor.fetchall()
                }
                
                # Performance trends
                cursor = conn.execute("""
                    SELECT DATE(timestamp) as date, metric_type, AVG(value) as avg_value
                    FROM performance_metrics 
                    WHERE timestamp >= ?
                    GROUP BY DATE(timestamp), metric_type
                    ORDER BY date
                """, (since_date,))
                
                performance_trends = {}
                for row in cursor.fetchall():
                    date, metric_type, avg_value = row
                    if metric_type not in performance_trends:
                        performance_trends[metric_type] = {}
                    performance_trends[metric_type][date] = round(avg_value, 3)
                
                return {
                    "avg_times": avg_times,
                    "performance_trends": performance_trends
                }
        except Exception:
            return {
                "avg_times": {},
                "performance_trends": {}
            }
    
    def get_popular_queries(self, days: int = 30, limit: int = 20) -> List[Dict[str, Any]]:
        """Get most popular queries"""
        try:
            since_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        JSON_EXTRACT(metadata, '$.query') as query,
                        COUNT(*) as frequency,
                        AVG(CAST(JSON_EXTRACT(metadata, '$.response_length') AS REAL)) as avg_response_length
                    FROM analytics_events 
                    WHERE timestamp >= ? 
                    AND event_type = 'query'
                    AND JSON_EXTRACT(metadata, '$.query') IS NOT NULL
                    GROUP BY JSON_EXTRACT(metadata, '$.query')
                    ORDER BY frequency DESC
                    LIMIT ?
                """, (since_date, limit))
                
                return [
                    {
                        "query": row[0],
                        "frequency": row[1],
                        "avg_response_length": round(row[2], 1) if row[2] else 0
                    }
                    for row in cursor.fetchall()
                ]
        except Exception:
            return []
    
    def export_analytics(self, days: int = 30) -> str:
        """Export analytics data as JSON"""
        try:
            data = {
                "export_date": datetime.now().isoformat(),
                "period_days": days,
                "usage_stats": self.get_usage_stats(days),
                "document_analytics": self.get_document_analytics(days),
                "performance_analytics": self.get_performance_analytics(days),
                "popular_queries": self.get_popular_queries(days)
            }
            
            return json.dumps(data, indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)


def create_analytics_visualizations(analytics: AnalyticsTracker, days: int = 30):
    """Create Plotly visualizations for analytics data"""
    
    if not PLOTLY_AVAILABLE:
        st.warning("📊 Plotly not installed. Install with: pip install plotly")
        st.info("Analytics data is still available in table format below.")
    
    # Get data
    usage_stats = analytics.get_usage_stats(days)
    doc_analytics = analytics.get_document_analytics(days)
    perf_analytics = analytics.get_performance_analytics(days)
    popular_queries = analytics.get_popular_queries(days, 10)
    
    st.markdown("### 📊 Usage Analytics")
    
    # Usage overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events", usage_stats["total_events"])
    with col2:
        st.metric("Unique Sessions", usage_stats["unique_sessions"])
    with col3:
        st.metric("Daily Avg", round(usage_stats["total_events"] / max(days, 1), 1))
    with col4:
        st.metric("Event Types", len(usage_stats["events_by_type"]))
    
    # Event types pie chart
    if usage_stats["events_by_type"] and PLOTLY_AVAILABLE:
        fig_pie = px.pie(
            values=list(usage_stats["events_by_type"].values()),
            names=list(usage_stats["events_by_type"].keys()),
            title="Events by Type"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    elif usage_stats["events_by_type"]:
        # Fallback to simple display
        st.markdown("**Events by Type:**")
        for event_type, count in usage_stats["events_by_type"].items():
            st.write(f"- {event_type}: {count}")
    
    # Daily activity
    if usage_stats["daily_activity"] and PLOTLY_AVAILABLE:
        dates = list(usage_stats["daily_activity"].keys())
        counts = list(usage_stats["daily_activity"].values())
        
        fig_line = px.line(
            x=dates, y=counts,
            title="Daily Activity Trend",
            labels={"x": "Date", "y": "Event Count"}
        )
        st.plotly_chart(fig_line, use_container_width=True)
    elif usage_stats["daily_activity"]:
        # Fallback to table
        st.markdown("**Daily Activity:**")
        daily_df = pd.DataFrame([
            {"Date": date, "Events": count}
            for date, count in usage_stats["daily_activity"].items()
        ])
        st.dataframe(daily_df, use_container_width=True)
    
    # Document analytics
    st.markdown("### 📚 Document Analytics")
    
    if doc_analytics["top_documents"]:
        doc_df = pd.DataFrame(doc_analytics["top_documents"])
        
        # Top documents bar chart
        if PLOTLY_AVAILABLE:
            fig_bar = px.bar(
                doc_df, 
                x="document", 
                y="engagement_count",
                title="Most Engaged Documents",
                color="avg_relevance",
                color_continuous_scale="viridis"
            )
            fig_bar.update_xaxes(tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Documents table
        st.dataframe(doc_df, use_container_width=True)
    
    # Performance analytics
    if perf_analytics["avg_times"]:
        st.markdown("### ⚡ Performance Metrics")
        
        perf_df = pd.DataFrame([
            {"Metric": metric, "Average Time (s)": data["avg"], "Count": data["count"]}
            for metric, data in perf_analytics["avg_times"].items()
        ])
        
        col1, col2 = st.columns(2)
        with col1:
            if PLOTLY_AVAILABLE:
                fig_perf = px.bar(
                    perf_df,
                    x="Metric",
                    y="Average Time (s)",
                    title="Average Response Times"
                )
                st.plotly_chart(fig_perf, use_container_width=True)
            else:
                st.markdown("**Performance Metrics:**")
                st.dataframe(perf_df, use_container_width=True)
        
        with col2:
            st.dataframe(perf_df, use_container_width=True)
    
    # Popular queries
    if popular_queries:
        st.markdown("### 🔍 Popular Queries")
        
        queries_df = pd.DataFrame(popular_queries)
        
        # Truncate long queries for display
        queries_df["query_short"] = queries_df["query"].apply(
            lambda x: x[:50] + "..." if len(x) > 50 else x
        )
        
        if PLOTLY_AVAILABLE:
            fig_queries = px.bar(
                queries_df.head(10),
                x="frequency",
                y="query_short",
                orientation="h",
                title="Top 10 Queries by Frequency"
            )
            fig_queries.update_layout(height=400)
            st.plotly_chart(fig_queries, use_container_width=True)
        
        # Full queries table
        st.dataframe(
            queries_df[["query", "frequency", "avg_response_length"]],
            use_container_width=True
        )


@st.cache_resource
def get_analytics_tracker() -> AnalyticsTracker:
    """Get cached analytics tracker instance"""
    db_path = os.getenv("ANALYTICS_DB_PATH", "analytics.db")
    return AnalyticsTracker(db_path)
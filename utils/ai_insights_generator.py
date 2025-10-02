"""
AI-powered insights generator using OpenAI API
Generates contextual financial insights for each category
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import json
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

from .text_utils import normalize_markdown_spacing


INSIGHT_FORMAT_VERSION = 2

# Load environment variables for local development
load_dotenv()


class AIInsightsGenerator:
    """Generate AI-powered financial insights using OpenAI"""
    
    def __init__(self, data_dir: Path):
        """Initialize the AI insights generator
        
        Args:
            data_dir: Path to the data directory for storing insights
        """
        self.data_dir = data_dir
        self.insights_dir = data_dir / "insights"
        self.insights_dir.mkdir(exist_ok=True)
        
        # Initialize OpenAI client - support both Streamlit secrets and env variables
        api_key = self._get_api_key()
        self.model = self._get_model()
        
        if not api_key or api_key == 'your-api-key-here':
            self.client = None
            self.enabled = False
        else:
            self.client = OpenAI(api_key=api_key)
            self.enabled = True
    
    def _get_api_key(self) -> Optional[str]:
        """Get OpenAI API key from Streamlit secrets or environment variables"""
        try:
            # Try Streamlit secrets first (for cloud deployment)
            if hasattr(st, 'secrets') and 'general' in st.secrets:
                return st.secrets.general.get('OPENAI_API_KEY')
        except Exception:
            pass
        
        # Fallback to environment variable (for local development)
        return os.getenv('OPENAI_API_KEY')
    
    def _get_model(self) -> str:
        """Get OpenAI model from Streamlit secrets or environment variables"""
        try:
            # Try Streamlit secrets first (for cloud deployment)
            if hasattr(st, 'secrets') and 'general' in st.secrets:
                return st.secrets.general.get('OPENAI_MODEL', 'gpt-4o-mini')
        except Exception:
            pass
        
        # Fallback to environment variable (for local development)
        return os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    
    def _get_insights_file_path(self, category_name: str) -> Path:
        """Get the file path for storing insights for a category"""
        # Convert category name to safe filename
        safe_name = category_name.lower().replace(' ', '_').replace('/', '_')
        return self.insights_dir / f"insights_{safe_name}.csv"
    
    def _load_cached_insight(self, category_name: str) -> Tuple[Optional[str], int]:
        """Load cached insight for a category if it exists"""
        file_path = self._get_insights_file_path(category_name)
        
        if not file_path.exists():
            return None, 0
        
        try:
            df = pd.read_csv(file_path)
            if len(df) > 0:
                row = df.iloc[0]
                version = row.get('format_version', 1)
                insight = row.get('insight', None)
                if version >= INSIGHT_FORMAT_VERSION:
                    return insight, version
                # Legacy cache - return insight for potential fallback but mark version outdated
                return insight, version
        except Exception as e:
            print(f"Error loading cached insight: {e}")
        
        return None, 0
    
    def _save_insight(self, category_name: str, insight: str):
        """Save generated insight to CSV file"""
        file_path = self._get_insights_file_path(category_name)
        
        try:
            df = pd.DataFrame({
                'category': [category_name],
                'insight': [insight],
                'generated_at': [datetime.now().isoformat()],
                'model': [self.model],
                'format_version': [INSIGHT_FORMAT_VERSION]
            })
            df.to_csv(file_path, index=False)
        except Exception as e:
            print(f"Error saving insight: {e}")
    
    def _prepare_data_context(self, category_name: str, data: pd.DataFrame) -> str:
        """Prepare financial data as context for the AI"""
        context = f"Category: {category_name}\n\n"
        context += "Financial Data:\n"
        context += data.to_string(index=False)
        
        return context
    
    def _create_prompt(self, category_name: str, data_context: str) -> str:
        """Create a detailed prompt for generating insights"""
        prompt = f"""You are a financial analyst providing insights on {category_name} data for Qualcomm (QCOM).

{data_context}

Craft a polished Markdown-formatted briefing using the exact structure below. Every bullet must reference specific data points (include the year and figure) and use bold formatting for key metrics. Keep bullets to one sentence so the overall response remains concise and skimmable.

### 📌 Key Observations
- ...
- ...

### 💰 Financial Health Assessment
- ...
- ...

### 📈 Year-over-Year Changes
- ...
- ...

### 🧭 Strategic Insights
- ...
- ...

### ✅ Actionable Takeaways
- ...
- ...

Replace the ellipses with substantive insights drawn from the data. Do not add extra sections, introductions, or conclusions outside of this template."""

        return prompt
    
    def generate_insight(self, category_name: str, data: pd.DataFrame, force_regenerate: bool = False) -> Dict[str, any]:
        """Generate AI-powered insight for a financial category
        
        Args:
            category_name: Name of the financial category
            data: DataFrame containing the financial data
            force_regenerate: If True, bypass cache and generate new insight
            
        Returns:
            Dict with 'insight' text and 'cached' boolean flag
        """
        # Check if AI is enabled
        if not self.enabled:
            return {
                'insight': "⚠️ AI insights are not configured. Please add your OpenAI API key to the .env file.",
                'cached': False,
                'error': True
            }
        
        # Check cache unless force regenerate
        legacy_insight: Optional[str] = None
        if not force_regenerate:
            cached_insight, cache_version = self._load_cached_insight(category_name)
            if cached_insight and cache_version >= INSIGHT_FORMAT_VERSION:
                cached_insight = normalize_markdown_spacing(cached_insight)
                return {
                    'insight': cached_insight,
                    'cached': True,
                    'error': False
                }
            elif cached_insight:
                legacy_insight = cached_insight
        
        # Generate new insight
        try:
            data_context = self._prepare_data_context(category_name, data)
            prompt = self._create_prompt(category_name, data_context)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert financial analyst specializing in technology companies. Provide clear, actionable insights based on financial data."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            insight = normalize_markdown_spacing(response.choices[0].message.content.strip())
            
            # Save to cache
            self._save_insight(category_name, insight)
            
            return {
                'insight': insight,
                'cached': False,
                'error': False
            }
            
        except Exception as e:
            error_msg = f"Error generating AI insight: {str(e)}"
            print(error_msg)
            
            # Try to return cached version if available
            if legacy_insight:
                cached_insight = normalize_markdown_spacing(legacy_insight)
                return {
                    'insight': cached_insight,
                    'cached': True,
                    'error': False,
                    'warning': 'Using cached insight due to API error'
                }
            
            return {
                'insight': f"Unable to generate AI insight. {str(e)}",
                'cached': False,
                'error': True
            }
    
    def clear_cache(self, category_name: Optional[str] = None):
        """Clear cached insights
        
        Args:
            category_name: If provided, clear only this category. Otherwise clear all.
        """
        if category_name:
            file_path = self._get_insights_file_path(category_name)
            if file_path.exists():
                file_path.unlink()
        else:
            # Clear all insight files
            for file_path in self.insights_dir.glob("insights_*.csv"):
                file_path.unlink()
    
    def get_all_cached_categories(self) -> list:
        """Get list of all categories with cached insights"""
        categories = []
        for file_path in self.insights_dir.glob("insights_*.csv"):
            # Extract category name from filename
            name = file_path.stem.replace('insights_', '').replace('_', ' ').title()
            categories.append(name)
        return categories

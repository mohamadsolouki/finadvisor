"""
AI-powered insights generator using OpenAI API
Generates contextual financial insights for each category
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
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
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        
        if not api_key or api_key == 'your-api-key-here':
            self.client = None
            self.enabled = False
        else:
            self.client = OpenAI(api_key=api_key)
            self.enabled = True
    
    def _get_insights_file_path(self, category_name: str) -> Path:
        """Get the file path for storing insights for a category"""
        # Convert category name to safe filename
        safe_name = category_name.lower().replace(' ', '_').replace('/', '_')
        return self.insights_dir / f"insights_{safe_name}.csv"
    
    def _load_cached_insight(self, category_name: str) -> Optional[str]:
        """Load cached insight for a category if it exists"""
        file_path = self._get_insights_file_path(category_name)
        
        if not file_path.exists():
            return None
        
        try:
            df = pd.read_csv(file_path)
            if len(df) > 0:
                return df.iloc[0]['insight']
        except Exception as e:
            print(f"Error loading cached insight: {e}")
        
        return None
    
    def _save_insight(self, category_name: str, insight: str):
        """Save generated insight to CSV file"""
        file_path = self._get_insights_file_path(category_name)
        
        try:
            df = pd.DataFrame({
                'category': [category_name],
                'insight': [insight],
                'generated_at': [datetime.now().isoformat()],
                'model': [self.model]
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

Based on the above financial data, provide a comprehensive, professional analysis with the following:

1. Key Observations: Identify the most significant trends, changes, or patterns in the data
2. Financial Health Assessment: Evaluate what these numbers indicate about the company's performance
3. Year-over-Year Changes: Highlight important changes between years and their implications
4. Strategic Insights: What do these metrics suggest about the company's strategy and position?
5. Actionable Takeaways: Provide 2-3 key insights that investors or stakeholders should focus on

Keep your response concise (3-5 sentences), professional, and focused on the most important insights. Use specific numbers from the data to support your analysis. Format as a single flowing paragraph that provides maximum value to readers."""

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
        if not force_regenerate:
            cached_insight = self._load_cached_insight(category_name)
            if cached_insight:
                return {
                    'insight': cached_insight,
                    'cached': True,
                    'error': False
                }
        
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
            
            insight = response.choices[0].message.content.strip()
            
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
            cached_insight = self._load_cached_insight(category_name)
            if cached_insight:
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

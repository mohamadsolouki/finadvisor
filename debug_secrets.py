"""
Debug utility to test Streamlit secrets access
Run this as a separate Streamlit app to debug secrets issues
"""

import streamlit as st
import os

st.title("Streamlit Secrets Debug Tool")

st.header("Environment Variables")
openai_env = os.getenv('OPENAI_API_KEY')
st.write(f"OPENAI_API_KEY from env: {'✅ Found' if openai_env else '❌ Not found'}")
if openai_env:
    st.write(f"Length: {len(openai_env)} characters")
    st.write(f"Starts with: {openai_env[:10]}..." if len(openai_env) > 10 else openai_env)

st.header("Streamlit Secrets")
st.write(f"st.secrets available: {'✅ Yes' if hasattr(st, 'secrets') else '❌ No'}")

if hasattr(st, 'secrets'):
    st.write(f"st.secrets type: {type(st.secrets)}")
    
    # Try to access secrets
    try:
        st.write("Available top-level keys:")
        if hasattr(st.secrets, '__dict__'):
            for key in st.secrets.__dict__.keys():
                st.write(f"  - {key}")
        
        # Try accessing general section
        if hasattr(st.secrets, 'general'):
            st.write("✅ st.secrets.general exists")
            if hasattr(st.secrets.general, 'keys'):
                st.write("Keys in general section:")
                for key in st.secrets.general.keys():
                    st.write(f"  - {key}")
            
            # Try to get API key
            try:
                api_key = st.secrets.general.get('OPENAI_API_KEY')
                st.write(f"OPENAI_API_KEY from secrets: {'✅ Found' if api_key else '❌ Not found'}")
                if api_key:
                    st.write(f"Length: {len(api_key)} characters")
                    st.write(f"Starts with: {api_key[:10]}..." if len(api_key) > 10 else api_key)
            except Exception as e:
                st.error(f"Error accessing OPENAI_API_KEY: {e}")
        else:
            st.write("❌ st.secrets.general does not exist")
            
            # Try alternative access methods
            try:
                api_key = getattr(st.secrets, 'OPENAI_API_KEY', None)
                st.write(f"Direct access OPENAI_API_KEY: {'✅ Found' if api_key else '❌ Not found'}")
            except Exception as e:
                st.error(f"Error with direct access: {e}")
                
    except Exception as e:
        st.error(f"Error accessing secrets: {e}")

st.header("Test OpenAI Client")
try:
    from openai import OpenAI
    
    # Test with environment variable
    if openai_env:
        try:
            client_env = OpenAI(api_key=openai_env)
            st.write("✅ OpenAI client initialized with env variable")
        except Exception as e:
            st.error(f"❌ Failed to initialize with env: {e}")
    
    # Test with secrets
    if hasattr(st, 'secrets'):
        try:
            if hasattr(st.secrets, 'general'):
                api_key_secrets = st.secrets.general.get('OPENAI_API_KEY')
            else:
                api_key_secrets = getattr(st.secrets, 'OPENAI_API_KEY', None)
                
            if api_key_secrets:
                client_secrets = OpenAI(api_key=api_key_secrets)
                st.write("✅ OpenAI client initialized with secrets")
            else:
                st.write("❌ No API key found in secrets")
        except Exception as e:
            st.error(f"❌ Failed to initialize with secrets: {e}")
            
except ImportError:
    st.error("OpenAI library not installed")
except Exception as e:
    st.error(f"Error testing OpenAI client: {e}")
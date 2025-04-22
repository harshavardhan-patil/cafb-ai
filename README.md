# CAFB AI Support System

The Capital Area Food Bank (CAFB) AI Support System is an intelligent customer service solution designed to enhance partner support operations by integrating with the existing Jira ticketing system. This system addresses key challenges in ticket management and customer service automation, providing real-time analytics and AI-powered assistance to improve response times and service quality.

## Video Demo

![CAFB_AI_Demo](reports/figures/CAFB_AI.mp4)

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------


## Problem Statement
CAFB partners with hundreds of nonprofits that rely on them for food distribution to their communities. The customer service team faces two major challenges:

1. **Limited Analytics Capabilities**: The Jira ticketing system provides basic data visualization but lacks advanced trend analysis and actionable insights that could drive strategic improvements.

2. **High Volume of Support Requests**: The manual handling of support tickets creates inefficiencies, with many common issues requiring repetitive responses from human agents.

## Solution Components

### 1. Customer Support Dashboard
An interactive analytics dashboard that provides comprehensive insights into support ticket data:

- **Trends Analysis**: Visualizes ticket volume across different time periods (yearly, monthly, daily)
- **Topic Modeling**: Identifies common issue categories and subcategories beyond Jira's predefined options
- **Heat Maps**: Shows correlations between different types of issues (e.g., missed deliveries and damaged goods)
- **Performance Metrics**: Tracks resolution times and team performance
- **Filtering Capabilities**: Allows filtering by priority, team member, region, and source

### 2. AI-Powered Support System
An intelligent chatbot interface that integrates with the existing Jira system:

- **Ticket Creation**: Guides partners through the process of creating new support tickets
- **Automated Responses**: Handles common inquiries without human intervention
- **Information Collection**: Gathers relevant details before escalating to human agents
- **Priority Management**: Identifies urgent issues and updates ticket priorities accordingly
- **Order Management**: Facilitates order modifications, cancellations, and delivery issue resolution
- **Knowledge Base Integration**: Provides partners with relevant information based on ticket context

## Key Features

### For New Issues
- Partner information collection
- Category and subcategory classification
- Automatic ticket creation in Jira
- Initial troubleshooting and information gathering

### For Existing Tickets
- Order modification support
- Delivery issue resolution
- Priority escalation
- Access to relevant knowledge base articles
- Sales order number tracking

### Technical Capabilities
- Seamless integration with Jira
- Real-time updates to ticket information
- Tool-based actions (closing tickets, updating priorities, adding order numbers)
- Context-aware responses based on ticket history and knowledge base

## Technology Stack
- **Frontend**: Streamlit for the web interface
- **Backend**: Python with LangChain for conversational AI
- **LLM Integration**: OpenAI API for natural language processing
- **Data Visualization**: Interactive dashboards for analytics
- **Ticketing System**: Integration with Jira API

## Installation and Setup

### Prerequisites
- Python 3.8+
- Streamlit
- OpenAI API key
- Jira access credentials

### Environment Variables
```
MODEL=<llm-model-name>
PROJECT_KEY=<jira-project-key>
OPENAI_API_KEY=<your-openai-api-key>
```

### Running the Application
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables
4. Run the application: `streamlit run support.py`

## Usage

### Creating a New Issue
1. Click on "New Issue"
2. Enter partner name
3. Select issue category and subcategory
4. Provide a brief description
5. Submit to create the ticket
6. Continue conversation with the AI assistant for further assistance

### Managing Existing Tickets
1. Click on "Existing Ticket"
2. Enter the Jira ticket ID/key
3. Interact with the AI assistant to:
   - Get updates on ticket status
   - Modify orders
   - Escalate priorities
   - Resolve delivery issues
   - Close tickets when resolved

## Benefits
- **Improved Efficiency**: Reduces manual workload for customer service representatives
- **Enhanced Insights**: Provides actionable data for strategic improvements
- **Faster Resolution**: Streamlines issue handling and prioritization
- **24/7 Support**: Offers round-the-clock assistance for partners
- **Continuous Learning**: System improves over time through interaction data

## Future Enhancements
- Advanced predictive analytics for proactive issue resolution
- Integration with additional communication channels (email, SMS)
- Enhanced natural language understanding for more complex queries
- Mobile application for on-the-go support
- Expanded knowledge base with automated updates


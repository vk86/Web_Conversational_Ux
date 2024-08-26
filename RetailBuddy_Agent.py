# Standard library imports
import json
import os
import sys
import time
from datetime import datetime
from re import A  # Note: Review the specific usage of 're.A'
from typing import Any, Dict, Optional, Type

# Azure specific libraries
from azure.search.documents.models import VectorizedQuery
from azure.search.documents import SearchClient
import openai
import json
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient


# Suppress specific warnings
import warnings
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)

# Environment Configuration
from dotenv import load_dotenv
load_dotenv()

# Flask Web Framework
from flask import Flask, jsonify, request, Response, send_from_directory
from flask_cors import CORS
from flask_restx import Api, Resource, fields
from werkzeug.utils import secure_filename

# Data Manipulation and Analysis
# import numpy as np
# import spacy
from sentence_transformers import SentenceTransformer, util

# HTTP Requests
import requests

# OpenAI and Retry Strategies
import openai
# from tenacity import retry, wait_random_exponential, stop_after_attempt

# Database and Object Relational Mapping (ORM)
# from sqlalchemy import Column, Integer, Text, DateTime, desc
# from sqlalchemy.orm import declarative_base

# Data Validation with Pydantic
from pydantic import BaseModel, Field

# Custom Langchain Library Imports
from langchain.agents import initialize_agent, Tool, AgentType
#from langchain.agents.agent_toolkits import AzureCognitiveServicesToolkit
from langchain.agents.agent_toolkits import AzureCognitiveServicesToolkit
from langchain.chat_models import AzureChatOpenAI
from langchain.chat_models import ChatOpenAI

#from langchain_openai import ChatOpenAI
from langchain.globals import set_debug
from openai import AzureOpenAI
# from langchain.llms import AzureOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory, SQLChatMessageHistory
from langchain.memory.chat_message_histories.sql import BaseMessageConverter
from langchain.prompts import MessagesPlaceholder
from langchain.schema import AIMessage, BaseMessage, ChatMessage, FunctionMessage, HumanMessage, SystemMessage
from langchain.tools import BaseTool, MoveFileTool, format_tool_to_openai_function

# Other Utilities
# from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import keys
from redis import Redis
from datetime import timedelta
from typing import Dict, Any

# Debugging and Development Utilities
# Print system path (Consider moving or removing this line from global scope)
print(sys.path)
set_debug(True)
# Used for FUNCTIONS LOGIC
# os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"] = keys.AZURE_SEARCH_SERVICE_ENDPOINT
# os.environ["AZURE_COGS_KEY"] = keys.AZURE_COGS_KEY
# os.environ["AZURE_COGS_ENDPOINT"] = keys.AZURE_COGS_ENDPOINT
# os.environ["AZURE_COGS_REGION"] = keys.AZURE_COGS_REGION

#from openai import OpenAI
#client = OpenAI()

#import openai
openai.api_key = "sk-proj-oR5JZOanzOyFyhjD0BmfT3BlbkFJGDhXpcZ0I2JaS7cmMsp1"
#OpenAI.api_key = "sk-proj-oR5JZOanzOyFyhjD0BmfT3BlbkFJGDhXpcZ0I2JaS7cmMsp1"

# The following variables from your .env file are used in this notebook
endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
credential = AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"]) if len(os.environ["AZURE_SEARCH_ADMIN_KEY"]) > 0 else DefaultAzureCredential()
index_name = os.environ["AZURE_SEARCH_INDEX"]

os.environ["AZURE_COGS_KEY"] = "f448f81e0f714f55969b1c7614938b1e"
os.environ["AZURE_COGS_ENDPOINT"] = "https://dev-azure-ai.cognitiveservices.azure.com/"
os.environ["AZURE_COGS_REGION"] = "eastus"

'''
def analyze_incidents_v1(description, incidents, summerize_text=True):
    """
    The function `analyze_incidents_v1` takes a description and a list of incidents as input, analyzes
    the similarity between the description and each incident's description, and returns relevant
    information, relevant incident numbers, and incident priorities.
    
    Args:
      description: The description parameter is a string that represents the description of the current
    incident that needs to be analyzed. It contains information about the current incident.
      incidents: The `incidents` parameter is a list of dictionaries, where each dictionary represents
    an incident. Each incident dictionary contains information about the incident, such as the incident
    number, description, priority, assigned to, comments, work notes, etc.
    
    Returns:
      The function `analyze_incidents_v1` returns three values: `relevantinfo`, `relevantincidents`, and
    `incident_priorities`.
    """
    model = SentenceTransformer('bert-base-nli-mean-tokens')  # Load the Sentence Transformers model

    relevantinfo = []
    relevantincidents = []
    incident_priorities = []
    incident_resolved_by = []
    historical_data = []
    similar_incidents = []
    
    similarity_threshold = 0.75  # Adjust the threshold based on your requirements

    # Embed the incident description into a fixed-dimensional vector
    incident_embedding = model.encode([description], convert_to_tensor=True)
    
    for incident in incidents:
        incident_number = incident.get('number', '')
        incident_desc = incident.get('description', '')
        incident_closenotes = incident.get('close_notes', '')
        incident_resolvedby = incident.get('assigned_to', '')
        if incident_resolvedby == "":
            incident_resolvedby = 'No Resolved By'
            
        incident_worknotes = incident.get('work_notes', '')
        incident_comments = incident.get('comments', '')
        incident_comments_and_worknotes = incident.get('comments_and_work_notes', '')
        incident_priority = incident.get('priority', '')
        
        # Compare the current incident with previously analyzed incidents
        issue_embedding = model.encode([incident_desc], convert_to_tensor=True)
        # print(f"Query Embeddings:{time.time()}")
        similarity_score = util.pytorch_cos_sim(incident_embedding, issue_embedding)
        
        if similarity_score.item() > similarity_threshold:
            if summerize_text:
                # Summarize the comments
                sim_incident = {}
                sim_incident['number'] = incident_number
                sim_incident['short_description'] = incident_priority
                sim_incident['description'] = incident_resolvedby
                sim_incident['close_notes'] = incident_closenotes
                sim_incident['work_notes'] = incident_worknotes
                sim_incident['comments'] = incident_comments
                sim_incident['comments_and_work_notes'] = incident_comments_and_worknotes
                similar_incidents.append(sim_incident)            
            historical_data.append(prep_incident_data(incident))
            relevantincidents.append(incident_number)
            incident_priorities.append(incident_priority)
            incident_resolved_by.append(incident_resolvedby)
    print(f"length of similar incidents: {len(similar_incidents)}")
    if (len(similar_incidents) > 0):
        summarize = summarize_incident_comments_v1(json.dumps(similar_incidents))
    else:
        summarize = "No similar incidents found"
    print(f"complete summary: {summarize}")
    relevantinfo.append(summarize)
    # combine relevantinfo, relevantincidents, incident_priorities into a dictionary
    relevantinfo = dict(zip(relevantincidents, relevantinfo))
       
    return relevantinfo, relevantincidents, incident_priorities, incident_resolved_by, historical_data


def get_entity_incidents_v1(description, assignment_group):
    """
    The function `get_entity_incidents_v1` takes in a incident description, and
    assignment group as parameters, and returns a list of incidents from ServiceNow that match the
    keywords extracted from the incident description.
    
    Args:
      description: The incident description that you want to extract keywords from.
      assignment_group: The `assignment_group` parameter is the name of the group responsible for
    handling the incidents. It is used to filter the search for incidents with the same key terms based
    on the specified assignment group.
    
    Returns:
      The function `get_entity_incidents_v1` returns a list of incidents that match the given criteria.
    The incidents are retrieved from the ServiceNow API based on the provided username, password,
    description, and assignment group. The function also prompts the user to provide relevant keywords
    for the incident description and uses those keywords to search for matching incidents. The function
    returns the raw JSON response from the ServiceNow API
    """
    
    url = f"{service_now_uri_main}/api/now/table/incident"
    
    ################################
    keyword_prompt = (
        f"""You are an AI assistant. Extract relevant keywords from the incident description provided within the <Incident Description> tags. (instruction)
        You need to generate a list of keywords specific to the incident description. These keywords will help in categorizing and searching for similar incidents in the ITSM tool.
        <Incident Description>
        {description}
        </Incident Description>
        Input Guideline:
        1.	Focus on extracting unique nouns and verbs specific to the incident.
        2.	Common words and stop words should be ignored.
        3.	The keywords should be in lowercase.
        4.	Arrange the keywords in alphabetical order.
        5.	Provide a maximum of 10 most relevant keywords.
        6.	If no keywords are found, return "No keywords found."
        Output Instructions/Guideline: Return only the list of keywords separated by spaces. Exclude any additional text.
        Keywords:"""
    )
    
    response = azureclient.chat.completions.create(
    model="gpt-4-32k",
    temperature=0.0,
    messages=[
        {"role": "system", "content": "You are useful AIassistant in incident management process"},
        {"role": "user", "content": keyword_prompt}
    ]
    )
    print("extracted response entities", response.choices[0].message.content)

    keywords = response.choices[0].message.content

    # grab keywords from the response
    entities = keywords    
    # return a max of 7 entities
    entities = [x.strip() for x in entities.split(",")]
    entities = entities[-7:]
    
    if not entities:
        print("No entities found.") 
    else:
        print("Entities found: ", entities)

    # Searching for incidents with the same key terms
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    auth = (service_now_username, service_now_password)

    ################################
    # number of incidents to return
    numberlimit = 20
    ################################
    
    params = {
        "sysparm_query": f"assignment_group.name={assignment_group}^state IN6,7^{','.join(entities)}^ORDERBYDESCsys_created_on",
        'sysparm_limit': numberlimit,  # Adjust the limit based on your requirements
        'sysparm_sort': 'sys_updated_on:desc',  # Sort incidents by the latest updated date in descending order
        'sysparm_display_value': True,
        'sysparm_exclude_reference_link': True,
    }
    # Response from ServiceNow on the search
    response = requests.get(url, headers=headers, auth=auth, params=params)

    #convert response to json
    incidents_raw = response.json().get('result')
    
    return incidents_raw

########## GENERATE RECOMMENDATION

def generate_recommendation(incident_number):
    """
    The `generate_recommendation` function generates a recommendation for a given incident number by
    analyzing past similar incidents, checking the current priority, and suggesting a priority change if
    necessary. It also includes relevant incident details, description, relevant tickets, and current
    SLA time in the recommendation.
    
    Args:
      incident_number: The incident number is a unique identifier for a specific incident. It is used to
    retrieve the details of the incident and generate a recommendation based on that information.
    
    Returns:
      The function `generate_recommendation` returns a dictionary with a key-value pair. The key is
    `'recommendation'` and the value is the generated recommendation.
    """
    
    print("Generating recommendation...")
    print("start generate recommendation step 1=", datetime.now().strftime("%H:%M:%S"))
    # To handle error if no incident not found.
    try:
        # Get description from incident 
        incidentjson = get_incident_details_v1(incident_number)
    except Exception as e:
        return {
            'recommendation': str(e)
        } 
    print("start generate recommendation step 2=", datetime.now().strftime("%H:%M:%S"))
    #extract description from incident json
    description = incidentjson.get("description", "")
    assignment_group = incidentjson.get("assignment_group", "")
    current_priority = incidentjson.get("priority", "")
    # get incidents via description with entities
    incidents_raw = get_entity_incidents_v1(description, assignment_group)
    print("start generate recommendation step 3=", datetime.now().strftime("%H:%M:%S"))
    
    #analyze incidents        
    relevantinfodata = analyze_incidents_v1(description, incidents_raw) #relinfo, reltickets
    print("start generate recommendation step 4=", datetime.now().strftime("%H:%M:%S"))

    #get the 3rd return from relevantinfodata and save as suggested_priority_list
    suggested_priority_list = relevantinfodata[2]
    #get the 4th return from relevantinfodata and save as resolved by
    suggested_resolved_by_list = relevantinfodata[3]
    #get the 2nd return from relevantinfodata and related incident numbers
    related_incidents = relevantinfodata[1]
    Historicaldata = relevantinfodata[4]
    print("related_incidents", related_incidents)
    # if relevantinfodata is empty, then relevantinfodata = "No relevant information found"
    if len(related_incidents) < 1:
        recommendation = "No similar incidents found."
    else:
        relevant_tickets = relevantinfodata[1]
        suggested_priority = get_suggestedpriority(suggested_priority_list, related_incidents, current_priority)   
        print("start generate recommendation step 5=", datetime.now().strftime("%H:%M:%S")) 
        current_sla = checkSLA(incident_number)
        print("start generate recommendation step 6=", datetime.now().strftime("%H:%M:%S"))
        # Create incident prompt for LLM. This will be used to generate the recommendation. 
        incident_prompt = (
            "As the AI Assistant, your task is to provide a highly detailed recommendation to resolve the incident, utilizing the incident description, work notes, comments, and other relevant incident details. Ensure your recommendation includes step-by-step solutions and scripts, if necessary. If a change in priority is necessary, indicate it at the end of the recommendation. Additionally, include information about who resolved similar issues based on past data, along with the current SLA time. Format your response using Markdown. You need to offer a comprehensive recommendation to address the incident, drawing from the incident description, work notes, comments, and relevant incident details. This recommendation aims to guide the user through resolving the incident effectively and efficiently based on below input data:"
            + "\n"
            + "Suggested Priority: "
            + suggested_priority
            + "\n"
            + "Relevant Incident details: "
            + str(relevantinfodata)
            + "\n"
            + "Incident Description: "
            + description
            + "\n"
            + "Relevant Tickets: "
            + str(relevant_tickets)
            + "\n"
            + "Current SLA: "
            + current_sla
            + "\n"
            + "Resolved By: "
            + str(suggested_resolved_by_list)
            + "\n"
            + "Input Guideline:/n"
            + "Utilize the provided incident details and relevant information to formulate the recommendation./n" 
            + "Include step-by-step solutions with detailed steps and scripts, if applicable./n"
            + "If a change in priority is recommended, mention it at the end of the recommendation./n"
            + "Incorporate information about individuals who resolved similar issues based on past data./n"
            + "Include the current SLA time in the recommendation./n"
            + "Format the response using Markdown for clarity and readability./n"
            + "Output Instructions/Guideline: Format your recommendation response in Markdown with the following attributes:/n"
            + "Resolution Steps: Detailed step-by-step solutions to resolve the incident/n"
            + "Order of the Steps: Sequential order of executing the recommended steps./n"
            + "Any Dependency: Mention any dependencies or prerequisites for executing specific steps/n"
            + "Check SLA: Reminder to check the current SLA time and adhere to it during incident resolution./n"
            + "Priority: If a change in priority is recommended, specify it at the end of the recommendation./n"
            + "Additional Details: Include any additional information or considerations relevant to resolving the incident./n"
            + "Resolved By: Provide information about individuals who resolved similar issues based on past data./n"
            + "Ensure clarity and completeness in your recommendation./n"
            + "Recommendation:"
        )

        response = azureclient.chat.completions.create(
        model="gpt-4-32k",
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are useful AIassistant in incident management process"},
            {"role": "user", "content": incident_prompt}
        ]
        )
        print("recommendation prompt", response)
        recommendation = response.choices[0].message.content
        print("start generate recommendation step 7=", datetime.now().strftime("%H:%M:%S"))
    return {
    'recommendation': recommendation
    }
    



########################################################################################################
# DATA PROCESSING AND VALIDATION
########################################################################################################
####### TOOL BUILDER
#######################################################################################################


####### TOOL INPUT FOR UPDATE ASSIGNED TO ########

class UpdateIncidentAssignedToInput(BaseModel):
    """Input for updating incident assigned-to field."""
    
    incident_number: str = Field(..., description="Incident number to be updated")
    assigned_to: str = Field(..., description="New assigned-to value for the incident")

####### TOOL BUILDER FOR UPDATE ASSIGNED TO ########

class UpdateIncidentAssignedToTool(BaseTool):
    name = "update_incident_assigned_to"
    description = "Useful for updating the 'assigned_to' field of an incident in ServiceNow. Do not use Current User in assigned_to field."

    def _run(self, incident_number: str, assigned_to: str):
        try:
            response = update_incident_info_assigned_to(service_now_uri_main, service_now_username, service_now_password, incident_number, assigned_to)
            return response 
        except Exception as e:
            # Log the exception or print it out. Consider using logging for production applications.
            print(f"An error occurred in UpdateIncidentAssignedToTool: {str(e)}")
            # Optionally, return a specific error message or re-raise the exception
            # depending on your application's error handling policy.
            return {"error": "An unexpected error occurred. Please try again later."}

    def _arun(self, incident_number: str, assigned_to: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = UpdateIncidentAssignedToInput
    
##########################
class UpdateIncidentStateInput(BaseModel):
    """Input for updating incident state field."""
   
    incident_number: str = Field(..., description="The Incident number for the ServiceNow incident which is to be updated. This data is required for the function to proceed.")
    state: str = Field(..., description="The 'state' field .of the incident in ServiceNow which is to be updated. The choices for 'State' are: New, In Progress, Pending, .Resolved, Closed. This data is required to proceed.")
    pending_reason: Optional[str] = Field(None, description="Pending reason in ServiceNow is mandatory field if a tickets 'state' is being changed to pending. The choices for Pending reason are: 1. Pending Change, 2. Pending Problem, 3. Pending 3rd Party, 4. Pending User, 5. Pending Duplicate. This is required data for the 'Pending' state. If this information is not provided, the tool will not be able to update the state to pending. If the information is needed and not provided, prompt the user to enter the appropriate pending reason. If user does not provide mandatory values for a required field, Do NOT update with generated values. Prompt the user to enter the appropriate Valuefor the field.")
    close_notes: Optional[str] = Field(None, description="Brief description of the resolution of the incident and any steps taken to resolve the incident. This is required only for changing 'state' to resolved. If this information is not provided, the tool will not be able to update the state to resolved. If the information is needed and not provided, prompt the user to enter the appropriate close notes. If user does not provide mandatory values for a required field, Do NOT update with generated values. Prompt the user to enter the appropriate Valuefor the field.")
    close_code: Optional[str] = Field(None, description="Brief 3 to 4 word description of the reason for closing the incident. This is required only for changing 'state' to closed. If this information is needed and not provided, the tool will not be able to update the state to closed. If the information is not provided prompt the user to enter the appropriate close code. If user does not provide mandatory values for a required field, Do NOT update with generated values. Prompt the user to enter the appropriate Valuefor the field.")
    work_notes: Optional[str] = Field(None, description="This is the work notes information for the incident. This is required only for changing 'state' to resolved. If this information is needed and not provided the tool will not be able to update the state to resolved. If the information is not provided, prompt the user to enter the appropriate work notes if they are asking to change state to 'resolved'. If user does not provide mandatory values for a required field, Do NOT update with generated values. Prompt the user to enter the appropriate Valuefor the field.")
    rfc: Optional[str] = Field(None, description="The Change Request number, or rfc must be provided only when 'state' is pending and 'pending reason' is requested to be changed to 'pending change'. If this information is not provided during these conditions, the tool will not be able to update the Pending reason to 'pending change'. If the information is needed and not provided prompt the user to enter the appropriate rfc. If user does not provide mandatory values for a required field, Do NOT update with generated values. Prompt the user to enter the appropriate Valuefor the field.")
    problem_id: Optional[str] = Field(None, description="The problem_id must be provided only when the incidents 'state' is pending and  'pending reason' is set to pending problem. If this information is not provided, the tool will not be able to update Pending reason to Pending problem. If the information is needed and not provided prompt the user to enter the appropriate problem_id. If user does not provide mandatory values for a required field, Do NOT update with generated values. Prompt the user to enter the appropriate Valuefor the field.")
    assigned_to: Optional[str] = Field(None, description="The assigned_to is required when the 'state' is being changed to In Progress. If user does not provide mandatory values for a required field, Do NOT update with generated values. Prompt the user to enter the appropriate Value for the field.")

class UpdateIncidentStateTool(BaseTool):
    name = "update_incident_info_state"
    description = "Useful for updating the 'state' field of an incident in ServiceNow. Depending on the state change being requested, the tool will prompt for additional information to complete the state change. The tool will validate the information provided and update the incident state accordingly."
 
    def _run(self, incident_number: str, state: str, assigned_to: str = None , pending_reason: str = None, close_notes: str = None, close_code: str = None, work_notes: str = None, rfc: str = None, problem_id: str = None):
        try:  
            if state == 'In Progress' and not assigned_to:
                assigned_to = input("Please enter the name of the person the incident is assigned to: ")
            print("Update Incident State Tool is running...")
            print(pending_reason)
            response = update_incident_info_state_v1(service_now_uri_main, service_now_username, service_now_password, incident_number, state, pending_reason, close_notes, close_code, work_notes, rfc,  problem_id, assigned_to)
            return response
        except Exception as e:
            print(f"An Error: {e}")
            return {"message": f"Error: {e}"}

class UpdateIncidentState_ToInprogressToolInput(BaseModel):
    """Input for updating 'state' field of an incident from 'New' or 'Pending' to 'In-Progress'"""
    
    incident_number: str = Field(..., description="Incident number to be updated")
    assigned_to: str = Field(..., description="assigned-to value for the incident")
    state: str = Field(..., description="default value is 'In Progress'")

class UpdateIncidentState_ToInprogressTool(BaseTool):
    name = "update_incident_state_to_inprogress"
    description = "Useful for updating the 'state' field of an incident in ServiceNow from 'New' or 'Pending' to 'In-Progress'"
    state="In Progress"
    def _run(self, incident_number: str,assigned_to: str, state: str,pending_reason: str = None, close_notes: str = None, close_code: str = None, work_notes: str = None, rfc: str = None, problem_id: str = None):
        try:
            response = update_incident_info_state_v1(service_now_uri_main, service_now_username, service_now_password, incident_number, state, pending_reason, close_notes, close_code, work_notes, rfc,  problem_id, assigned_to)
            print("State Change response",response)
            return response 
        except Exception as e:
            # Log the exception or print it out. Consider using logging for production applications.
            #print(f"An error occurred in UpdateIncidentState_ToInprogressTool: {str(e)}")
            # Optionally, return a specific error message or re-raise the exception
            # depending on your application's error handling policy.
            return {"error": "An unexpected error occurred. Please try again later."}

    def _arun(self, incident_number: str, assigned_to: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = UpdateIncidentState_ToInprogressToolInput

class UpdateIncidentState_ToResolvedToolInput(BaseModel):
    """Input for updating 'state' field of an incident from 'In-Progress' or 'Pending' to 'Resolved'"""
    
    incident_number: str = Field(..., description="Incident number to be updated")    
    close_notes: str = Field(..., description="close_notes value for the incident")
    close_code: str = Field(..., description="close_code value for the incident")
    work_notes: str = Field(..., description="work_notes value for the incident")
    state: str = Field(..., description="default value is 'Resolved'")

class UpdateIncidentState_ToResolvedTool(BaseTool):
    name = "update_incident_state_to_resolved"
    description = "Useful for updating the 'state' field of an incident in ServiceNow from 'In-Progress' or 'Pending' to 'Resolved'"
    state="Resolved"
    def _run(self, incident_number: str,close_notes: str, close_code: str, work_notes: str, state: str,assigned_to: str=None, pending_reason: str = None,  rfc: str = None, problem_id: str = None):
        try:
            response = update_incident_info_state_v1(service_now_uri_main, service_now_username, service_now_password, incident_number, state, pending_reason, close_notes, close_code, work_notes, rfc,  problem_id, assigned_to)
            print("State Change response",response)
            return response 
        except Exception as e:
            # Log the exception or print it out. Consider using logging for production applications.
            #print(f"An error occurred in UpdateIncidentState_ToResolvedTool: {str(e)}")
            # Optionally, return a specific error message or re-raise the exception
            # depending on your application's error handling policy.
            return {"error": "An unexpected error occurred. Please try again later."}

    def _arun(self, incident_number: str, close_notes: str, close_code: str, work_notes: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = UpdateIncidentState_ToResolvedToolInput

class UpdateIncidentState_ToPendingToolInput(BaseModel):
    """Input for updating 'state' field from 'In Progress' to 'Pending' of an incident."""
    incident_number: str = Field(..., description="Incident number to be updated")    
    pending_reason: str = Field(..., description="pending_reason value for the incident")
    state: str = Field(..., description="default value is 'Pending'")

class UpdateIncidentState_ToPendingTool(BaseTool):
    name = "update_incident_state_to_pending"
    description = "updating 'state' field from 'In Progress' to 'Pending' of an incident. When a user asks you to 'update state as Pending' provide a list to select valid pending reason as follows:\n\n1. Pending 3rd Party\n2. Pending User\n3. Pending Duplicate."
    state="Pending"
    def _run(self, incident_number: str,pending_reason: str, state: str, close_notes: str = None, close_code: str = None, work_notes: str = None, assigned_to: str=None, rfc: str = None, problem_id: str = None):
        try:
            response = update_incident_info_state_v1(service_now_uri_main, service_now_username, service_now_password, incident_number, state, pending_reason, close_notes, close_code, work_notes, rfc,  problem_id, assigned_to)
            print("State Change response",response)
            return response 
        except Exception as e:
            # Log the exception or print it out. Consider using logging for production applications.
            #print(f"An error occurred in UpdateIncidentState_ToPendingTool: {str(e)}")
            # Optionally, return a specific error message or re-raise the exception
            # depending on your application's error handling policy.
            return {"error": "An unexpected error occurred. Please try again later."}

    def _arun(self, incident_number: str, pending_reason: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = UpdateIncidentState_ToPendingToolInput

'''

def azure_search_query_draft(userquery, summarize_text=True):
    # Pure Vector Search
    #query = "Marketside Roasted Red Pepper Hummus"
    query = userquery
    embedding = openai.embeddings.create(input=query, model="text-embedding-ada-002").data[0].embedding
    vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=3, fields="prodnameVector")

    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

    results = search_client.search(  
        search_text=None,  
        vector_queries= [vector_query],
        select=["PRODUCT_NAME", "PRODUCT_DESCRIPTION", "CATEGORY","BRAND","PRICE_CURRENT","PROMOTION"]
    )  
    
    search_results = []
    for result in results:  
        print(f"PRODUCT_NAME: {result['PRODUCT_NAME']}")
        print(f"BRAND: {result['BRAND']}")
        print(f"Score: {result['@search.score']}")
        print(f"PRODUCT_DESCRIPTION: {result['PRODUCT_DESCRIPTION']}")  
        print(f"CATEGORY: {result['CATEGORY']}\n")
        print(f"PRICE_CURRENT: {result['PRICE_CURRENT']}")
        print(f"PROMOTION: {result['PROMOTION']}")
        print("\n")
    
    #search_results = []
    #for result in results:  
        search_results.append({
            'PRODUCT_NAME': result['PRODUCT_NAME'],
            'BRAND': result['BRAND'],
            '@search.score': result['@search.score'],
            'PRODUCT_DESCRIPTION': result['PRODUCT_DESCRIPTION'],
            'CATEGORY': result['CATEGORY'],
            'PRICE_CURRENT': result['PRICE_CURRENT'],
            'PROMOTION': result.get('PROMOTION', '')  # Handle cases where PROMOTION might be missing
        })

    return search_results

def azure_search_query_v1(search_query, search_index, search_fields, search_filter=None, search_orderby=None, search_top=5):
    """
    The function `azure_search_query_v1` performs a search query using the Azure Cognitive Search service.
    
    Args:
      search_query: The search query is a string that represents the query to be executed.
      search_index: The search index is a string that represents the index to search within.
      search_fields: The search fields is a list of strings that represent the fields to search within.
      search_filter: The search filter is a string that represents the filter to apply to the search query.
      search_orderby: The search orderby is a string that represents the order to apply to the search results.
      search_top: The search top is an integer that represents the number of results to return.
    
    Returns:
      The function `azure_search_query_v1` returns a dictionary containing the search results.
    """
    
    # Define the search parameters
    search_params = {
        "search": search_query,
        "searchFields": ",".join(search_fields),
        "queryType": "full",
        "top": search_top
    }
    
    # Add the filter parameter if provided
    if search_filter:
        search_params["filter"] = search_filter
    
    # Add the orderby parameter if provided
    if search_orderby:
        search_params["orderby"] = search_orderby
    
    # Perform the search query
    #search_results = azure_search_client.search(search_index, **search_params)
    #return search_results



def generate_product_recommendation_prompt(user_query, search_results):
    product_prompt = (
        "As the AI Assistant, your task is to provide a highly detailed product recommendation based on the user's query and the retrieved product details. Utilize the product descriptions, prices, promotional offers, and other relevant product details to formulate your recommendation. Ensure your recommendation includes a comparison of the products, highlighting the best option in terms of price, promotions, and user interest. Format your response using Markdown. Your recommendation should guide the user effectively and efficiently to make an informed purchasing decision based on the input data below:"
        + "\n\n"
        + "User Query: "
        + f"{user_query}"
        + "\n\n"
        + "Retrieved Products:\n"
    )

    for i, result in enumerate(search_results, start=1):
        product_prompt += (
            f"{i}. **PRODUCT_NAME**: {result['PRODUCT_NAME']}\n"
            f"   - **BRAND**: {result['BRAND']}\n"
            f"   - **Score**: {result['@search.score']}\n"
            f"   - **PRODUCT_DESCRIPTION**: {result['PRODUCT_DESCRIPTION']}\n"
            f"   - **CATEGORY**: {result['CATEGORY']}\n"
            f"   - **PRICE_CURRENT**: ${result['PRICE_CURRENT']}\n"
            f"   - **PROMOTION**: {result['PROMOTION']}\n\n"
        )

    product_prompt += (
        "Input Guidelines:\n"
        "- Utilize the provided product details and relevant information to formulate the recommendation.\n"
        "- Include a comparison of the products considering price, promotions, and product descriptions.\n"
        "- Recommend the best product for the user to buy, with a clear and concise explanation for your choice.\n"
        "- Format the response using Markdown for clarity and readability.\n\n"
        "Output Instructions/Guidelines:\n"
        "- **Top Recommended Product**: Clearly state the recommended product and brand.\n"
        "- **Comparison**: Provide a comparison of the products based on the given details.\n"
        "- **Reasoning**: Explain the reasoning behind your recommendation.\n"
        "- **Pricing**: Provide pricing.\n"
        "- **Additional Details**: Include any additional information that might help the user make an informed decision.\n"
        "- **Suggest Alternatives**: List other products and brands you have considered as suggested alternatives.\n"
        "- Ensure clarity and completeness in your recommendation.\n\n"
        "Recommendation:"
    )

    return product_prompt

def generate_product_recommendations(userquery, summarize_text=True):
    """
    The function takes a user question as input with product details and perform search in vector database
    
    """
    print("Generating product recommendation...")
    response = azure_search_query_draft(userquery)
    print("Step1")
    print("Vector search response:", response)
    product_prompt = generate_product_recommendation_prompt(userquery, response)
    print("Step2")
    print("Recommendation prompt", product_prompt)
    # Step 3: Call OpenAI Chat Completions API
    final_response = openai.chat.completions.create(
        #model="gpt-3.5-turbo-0125",
        model="gpt-4o-2024-05-13",
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant in the product recommendation process.Do not make up responses.Use data only from the input and do not hallucinate"},
            {"role": "user", "content": product_prompt}
        ]
    )
    
    print("Final Recommendation", final_response)
    recommendation = final_response.choices[0].message.content
    print("Recommendation generated at =", datetime.now().strftime("%H:%M:%S"))
    
    return recommendation
    
    #return {
    #    'recommendation': recommendation
    #}

def generate_product_search_prompt(user_query, search_results):
    product_prompt = (
        "As the AI Assistant, your task is to search for products based on the user's query and provide a detailed list of the retrieved products. Utilize the product descriptions, prices, promotional offers, and other relevant product details to present the results. Ensure the response is formatted using Markdown and presented in a tabular or list format for clarity and readability."
        + "\n\n"
        + "User Query: "
        + f"{user_query}"
        + "\n\n"
        + "Retrieved Products:\n"
    )

    product_prompt += (
        "| **#** | **Product Name** | **Brand** | **Score** | **Description** | **Category** | **Price** | **Promotion** |\n"
        "|-------|------------------|-----------|------------|-----------------|--------------|-----------|---------------|\n"
    )

    for i, result in enumerate(search_results, start=1):
        product_prompt += (
            f"| {i} | {result['PRODUCT_NAME']} | {result['BRAND']} | {result['@search.score']} | {result['PRODUCT_DESCRIPTION']} | {result['CATEGORY']} | ${result['PRICE_CURRENT']} | {result['PROMOTION']} |\n"
        )

    product_prompt += (
        "\n"
        "Input Guidelines:\n"
        "- Utilize the provided product details and relevant information to formulate the search results.\n"
        "- Present the results in a clear and concise manner using a table or list format.\n\n"
        "Output Instructions/Guidelines:\n"
        "- **Product List**: Provide a list of products with their details.\n"
        "- **Formatting**: Ensure the response is formatted using Markdown for clarity and readability.\n"
        "- **Additional Details**: Include any additional information that might help the user make an informed decision.\n\n"
        "Search Results:"
    )

    return product_prompt


def search_product_details(item_tosearch, summarize_text=True):
    """
    The function takes a user question as input with product details and perform search in vector database
    
    """
    print("Retrieve product details...")
    response = azure_search_query_draft(item_tosearch)
    print("Step1")
    product_prompt = generate_product_search_prompt(item_tosearch, response)
    print("Step2")
    
    # Step 3: Call OpenAI Chat Completions API
    final_response = openai.chat.completions.create(
        #model="gpt-3.5-turbo-0125",
        model="gpt-4o-2024-05-13",
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant in the product recommendation process"},
            {"role": "user", "content": product_prompt}
        ]
    )
    
    print("Recommendation prompt", final_response)
    recommendation = final_response.choices[0].message.content
    print("Recommendation generated at =", datetime.now().strftime("%H:%M:%S"))
    
    return recommendation
    
    #return {
    #    'recommendation': recommendation
    #}

def addto_cart(itemtoadd):
    import requests
    from requests.auth import HTTPBasicAuth
    import json

    # Define the URL
    url = 'http://localhost:8765/carts'

    # Define the payload
    payload = {
        "userId": 5,
        "date": "2020-02-03",
        "products": [
            {"productId": 5, "quantity": 1},
            {"productId": 1, "quantity": 5}
        ]
    }

    # Define your username and password for HTTP Basic Auth
    username = 'your_username'
    password = 'your_password'

    # Perform the POST request with HTTP Basic Auth
    response = requests.post(url, data=json.dumps(payload), headers={'Content-Type': 'application/json'}, auth=HTTPBasicAuth(username, password))

    # Check if the request was successful
    if response.status_code == 200:
        # Print the JSON response
        print(response.json())
    else:
        print(f"Failed to add item to cart: {response.status_code}")
        print(response.text)
    return response.json()
####### TOOL BUILDER FOR UPDATE ASSIGNED TO ########
#-----------------------------------------------------------------------------------------
class ProductRecommendationInput(BaseModel):
    """Input for generating product recommendation based on user query."""
    
    user_query: str = Field(..., description="User query related to product recommendation")

class ProductRecommendationTool(BaseTool):
    name = "generate_product_recommendation"
    description = "Useful for Generating product recommendation based on user input.Use this tool if user is specifically for product recommendation."

    def _run(self, user_query: str):
        try:
            response = generate_product_recommendations(user_query)
            return response 
        except Exception as e:
            print(f"An error occurred ")
            # Optionally, return a specific error message or re-raise the exception
            # depending on your application's error handling policy.
            return {"error": "An unexpected error occurred. Please try again later."}

    def _arun(self, incident_number: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = ProductRecommendationInput
#---------------------------------------------
class ProductSearchInput(BaseModel):
    """Input for getting product details based on user question."""
    
    itemtosearch: str = Field(..., description="Item or product the user is trying to search for details.")

class ProductSearchTool(BaseTool):
    name = "product_search"
    description = "Useful for getting product details and information based on user question.Use this tool if user is specifically for product details and response should be in tabular format."

    def _run(self, itemtosearch: str):
        try:
            response = search_product_details(itemtosearch)
            return response 
        except Exception as e:
            print(f"An error occurred ")
            # Optionally, return a specific error message or re-raise the exception
            # depending on your application's error handling policy.
            return {"error": "An unexpected error occurred. Please try again later."}

    def _arun(self, itemtosearch: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = ProductSearchInput
#---------------------------------------------
class AddtoCartToolInput(BaseModel):
    """Input for adding a product to usercart."""
    
    itemtoadd: str = Field(..., description="Item or product the user is trying to add to cart.")

class AddtoCartTool(BaseTool):
    name = "addTo_cart"
    description = "Useful for getting product details and information based on user question.Use this tool if user is specifically for product details and response should be in tabular format."

    def _run(self, itemtoadd: str):
        try:
            response = addto_cart(itemtoadd)
            return response 
        except Exception as e:
            print(f"An error occurred ")
            # Optionally, return a specific error message or re-raise the exception
            # depending on your application's error handling policy.
            return {"error": "An unexpected error occurred. Please try again later."}

    def _arun(self, itemtosearch: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = AddtoCartToolInput
#---------------------------------------------------------------------------------
class GenericFallbackInput(BaseModel):
    """Input for generic fallback responses."""
    
    user_question: str = Field(..., description="The user's question that couldn't be processed by other tools")
    
class GenericFallbackTool(BaseTool):
    name = "generic_fallback"
    description = "This tool provides a generic response for queries that cannot be processed by specific incident management tools."

    def _run(self, user_question: str):
        """
        Generates a generic response indicating that the user's question could not be processed
        with the available incident management tools and provides guidance for rephrasing the query.
        """
        
        # Log the unprocessed query for analysis and improvement purposes
        print(f"Received an unprocessable query: {user_question}")
        
        # Define a generic response
        response = (
            "I'm sorry, but I couldn't find a specific tool to help with your query. "
            "Could you please provide more details or rephrase your question? "
            "For instance, you can ask about specific incidents."
        )
        
        return response

    def _arun(self, user_question: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = GenericFallbackInput
#######################################################
# IN MEMORY DEFINITIONS
#######################################################
def store_chat_history(user_id, message):
    # Store the user's message in Redis, using a list structure
    ##redis_client.lpush(f'chat_history:{user_id}', json.dumps(message))

    # Optional: Trim the history to the last 10 messages to conserve space
    ##redis_client.ltrim(f'chat_history:{user_id}', 0, 9)
    print("none")

def get_chat_history(user_id):
    # Retrieve the last 10 messages from the history for the user
    ##history = redis_client.lrange(f'chat_history:{user_id}', 0, 9)
    
    # Convert from JSON strings back to dictionaries
    ##history = [json.loads(message) for message in history]
    #return history
    print("none")
##################
    
################################################################################
# MAIN
#################################################################################  
    
app = Flask(__name__)

# Initialize Flask-RESTx
api = Api(app, version='1.5', title='Retail Agent Orchestrator - API', description='Retail Agent Orchestrator API', doc='/api/doc')

chat_model = api.model('Chat', {
    'userquestion': fields.String(description='The user question'),
    'userinfo': fields.String(description='The user information'),
    'userAD': fields.String(description='The user Active Directory information')
})

toolkit = AzureCognitiveServicesToolkit()

toolkits = toolkit.get_tools()

print("Toolkits: ", toolkits)


tools = [ProductRecommendationTool(),ProductSearchTool(),AddtoCartTool(), GenericFallbackTool()]

'''
system_message = SystemMessage(content="""
You are an AI assistant dedicated to supporting retail customer experience processes. Your primary role is to assist customers by leveraging a suite of specialized tools. Please adhere to the following guidelines when responding to user queries:

1. **Action Simulation Prohibition**: As a language model connected to a set of tools that can make function calls to perform actions, you should only state or imply that a function call has been made if you have explicitly been provided knowledge that this has taken place. If you interpret a message as relevant to one of your function calls and make a function call, proceed accordingly with an output message that reflects the function call you have made. Under no circumstances should you state or imply that a function call has been made unless this is explicitly confirmed. Your output message should not allude to a change in the system being made unless it has actually occurred.

2. **Tool Utilization**: For questions related to customer experience and support, utilize the available tools to provide accurate and precise responses. Each tool is designed to handle specific aspects of customer support, such as checking order status, processing returns, or providing product information.

    2.1 If GetOrdersByCustomerTool is used, respond back in tabular format. Summarize total orders and status notes at the end.

3. **Generic Interactions**: For generic greetings or inquiries (e.g., "Hi", "Hello", "How are you?"), respond in a friendly and engaging manner. However, keep the responses brief and professional.

4. **Beyond Tool Scope**: If a user's question pertains to customer experience but does not match the capabilities of the existing tools, kindly advise the user to rephrase their question or specify their request. Example response: "I'm here to assist with your retail experience. Could you please provide more details about your request or the assistance you need?"

5. **Avoid Hallucination**: In cases where a query falls outside our supported functionalities and no tool can adequately address it, do not fabricate responses or provide speculative answers. Instead, respectfully inform the user that the request cannot be processed as is. Example response: "I'm sorry, but I can't provide a specific answer to that query. Can you please rephrase your question or ask about something else related to your retail experience?"

By following these instructions, ensure that your interactions are helpful, relevant, and grounded in the capabilities of our customer experience support system.
""")
'''
system_message = SystemMessage(content="""
You are an AI assistant supporting retail customer experiences. Follow these guidelines when responding:

1. **Action Simulation**: Only state a function call has been made if you know it has. Do not imply changes unless confirmed.
2. **Tool Output**: Provide tool output exactly as it is.
3. **Generic Interactions**: Respond briefly and professionally to greetings and general inquiries.Mention the user name in the response for greetings , hi, hello etc.
4. **Beyond Tool Scope**: If the user's request is unclear or beyond tool capabilities, ask for more details. Example: "Please provide more details about your request."
5. **Avoid Hallucination**: If you can't address a query, do not make up answers. Example: "I'm sorry, but I can't provide a specific answer to that query. Can you please rephrase your question?"

Ensure your interactions are helpful, relevant, and based on the tools' capabilities.
""")
#agent_kwargs = {
#    "system_message": system_message,
#    "max_iterations": 30
#}

agent_kwargs = {
    "max_iterations": 30
}

#llm = AzureChatOpenAI(
#    deployment_name="gpt-4-32k",
#    model_name="gpt-4-32k",
#    temperature=0
#)

llm = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0.0,api_key=keys.OPENAI_API_KEY)

####Initialize Redis
# Initialize Redis connection
#redis_client = Redis(
#    host=keys.AZURE_REDIS_HOST, 
#    port=6380, 
#    password=keys.AZURE_REDIS_PASSWORD, 
#    ssl=True,
#    ssl_cert_reqs=None
#)             

@api.route('/chat')
class Chat(Resource):
    @api.expect(chat_model)
    def post(self):
        try:
            userquestion = request.json['userquestion']

            userinfo = request.json['userinfo']
            userAD = request.json['userAD']
                       
            session_id = userinfo

            print("User Info: ", userinfo),
            print("User AD:",userAD)
            user_chat_history = get_chat_history(userAD)
            
            print("UserChathistory:",user_chat_history)
            
            #### MEMORY DB ######
            message_history = "Hi"
            #message_history = RedisChatMessageHistory(
            #url="redis://:HhhknNtwEDlGLkwV8ZOtAXcsC5EF55tMVAzCaGpkI6o=@gadm-aiops-redis.redis.cache.windows.net:6379", ttl=300, session_id=session_id
            #)
            #memory_key="memory", chat_memory=message_history, input_key="input", output_key="output", return_messages=True
            
            memory = ConversationBufferMemory(
                memory_key="memory",  input_key="input", output_key="output", return_messages=True
            )
            #### SECURITY ############
            # Create a list of people who can use the chatbot
            access_control_list = []
            
            # Load the access_control_list from auth.txt file
            with open("auth.txt", "r") as f:
                 for line in f:
                     access_control_list.append(line.strip())
            
            # Function to check if the user is in the access control list
            def is_user_authorized(user_info):
                return user_info in access_control_list

            # Check if the user is authorized
            if is_user_authorized(userinfo):
                # Allow the user to use the chatbot
                print("Access granted. You can use the chatbot.")
                              
                if "," in userinfo:
                    userinfo = userinfo.split(",")
                    userinfo = userinfo[1] + " " + userinfo[0]
                else:
                    userinfo = userinfo
                
                print("current user",userinfo)
                
                request.current_user = userinfo
                #userquestion = f"My username is {userinfo}. "+ userquestion
                userquestion = f"{userinfo}:"+ userquestion
                ###### AGENT ############
                open_ai_agent = initialize_agent(toolkits+tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True, memory=memory,agent_kwargs=agent_kwargs, return_intermediate_steps=True)

                memory.chat_memory.add_user_message(f"input: my name is {userinfo}")
                print("MEMORY: ", memory.load_memory_variables({}))
                print("agent_kwargs:",agent_kwargs)

                
                response=""
                # Check if the current question was asked in the last 10 questions
                if False:
                # if userquestion in [message['text'] for message in user_chat_history]:
                    # Find the corresponding response
                    for message in reversed(user_chat_history):
                        if message['text'] == userquestion:
                            print("Found in Cache")
                            return {"response": message['response']}
                else :
                    # Use this prompt with the LLM
                    #response = open_ai_agent(userquestion,userinfo)
                    print("In Final Response")
                    response = open_ai_agent(userquestion,userinfo)
                response = open_ai_agent(userquestion,userinfo)
                
                # print to console
                print("Final Response:",response)
                
                ############################################
                
                # Ensure the response contains the 'output' key
                if 'output' in response:
                    ai_response = response['output']
                    # Store the new question and response in Azure Redis Cache
                    store_chat_history(userAD, {'text': userquestion, 'response': ai_response})
                else:
                    ai_response = "No valid response from the AI."

                # Return only the 'output' key in the response
                return {"response": ai_response}
               
            else:
                # Deny access and provide an error message
                error_message = "Access denied. You are not authorized to use CoPilot."
                # Save the userinfo to a file
                with open("access_denied.txt", "a") as f:
                    f.write(f"User: {userinfo} Denied Access.\n")
                print(error_message)
                
                # Return an error response
                return {"response": error_message} # 403 indicates "Forbidden" access
                            
            
        except Exception as e:
            return {"error": str(e)}

if __name__ == '__main__':
    # Define the custom IP address and port
    #custom_ip = '52.252.230.72'  # Set to the desired IP address (e.g., '127.0.0.1' for localhost)
    # custom_ip = '10.0.0.6'
    custom_ip = '127.0.0.1'
    custom_port = 5000   # Set to the desired port number (e.g., 8080)
    # Run the Flask app with custom IP and port
    app.run(debug=True, host=custom_ip, port=custom_port)
import os
import json
from openai import AzureOpenAI

versions = {'0125':{'resource_name':'0125-Preview', 'deployment_name':'0125-Preview'},
            '1106':{'resource_name':'MMResearch', 'deployment_name':'gpt-4-1106-Preview'},
            '0613':{'resource_name':'0613', 'deployment_name':'0613'}}

id='0613'
no='new'

client = AzureOpenAI(
  azure_endpoint = os.getenv(f"AZURE_OPENAI_ENDPOINT_{id}"), 
  api_key=os.getenv(f"AZURE_OPENAI_API_KEY_{id}"),  
  api_version="2024-02-01"
)

print(os.getenv(f"AZURE_OPENAI_ENDPOINT_{id}"))

# load json dict
with open(f'all_tasks_dict_{no}.json', 'r') as f:
    story_dict = json.load(f)

gpt_responses = {}

for k in story_dict:
    gpt_responses[k] = {}
    story_1 = story_dict[k]['Story_1']
    story_a = story_dict[k]['Story_A']
    story_b = story_dict[k]['Story_B']
    prompt_1 = f"Consider the following story:\n\nStory 1: {story_1}\n\nNow consider two more stories:\n\nStory A: {story_a}\n\nStory B: {story_b}\n\nWhich of Story A and Story B is a better analogy to Story 1? Is the best answer Story A, Story B, or both are equally analogous?"
    prompt_2 = f"Consider the following story:\n\nStory 1: {story_1}\n\nNow consider two more stories:\n\nStory A: {story_b}\n\nStory B: {story_a}\n\nWhich of Story A and Story B is a better analogy to Story 1? Is the best answer Story A, Story B, or both are equally analogous?"
    response = client.chat.completions.create(
        model= versions[id]['deployment_name'],#"gpt-4-1106-Preview", # model = "deployment_name".
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_1},
            ]
    )
    gpt_responses[k]['order_1'] = response.choices[0].message.content
    
    response = client.chat.completions.create(
        model= versions[id]['deployment_name'], # model = "deployment_name".
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_2},
            ]
    )
    gpt_responses[k]['order_2'] = response.choices[0].message.content

json_string = json.dumps(gpt_responses, indent=2)

with open(f'gpt_results/gpt_{id}_responses_dict_{no}.json', 'w') as json_f:
    json_f.write(json_string)
    
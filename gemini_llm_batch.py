import pandas as pd
import json
import os
import math 
import base64
import os
import subprocess
from typing import Optional
import shlex
import json
import re
import argparse
# from utils import upload_with_subprocess, download_from_gcs, quick_response_check, read_jsonl
import re
import time
from datetime import datetime
import ast

parser = argparse.ArgumentParser(description='Process CSV file with Gemini LLM batch processing')
parser.add_argument('--newcsvpath', type=str, required=True, help='Path to the input CSV file')
parser.add_argument('--text_col', type=str, default='text', help='Name of the text column in the CSV file (default: text)')
parser.add_argument('--location_guidance', action='store_true', help='Use location guidance')

args = parser.parse_args()
text_col = args.text_col


version = "flash"
model = f"gemini-2.5-{version}"

def upload_with_subprocess(local_file, bucket, destination):
    """Upload file to GCS bucket using gsutil command line via Python"""

    gcs_uri = f"gs://{bucket}/{destination}"
    
    # Use shlex.quote to safely escape paths with spaces
    safe_gcs_uri = shlex.quote(gcs_uri)
    safe_local_path = shlex.quote(local_file)
    command = f"gsutil cp  {safe_local_path} {safe_gcs_uri}"
    # command = ["gsutil", "cp", safe_local_path, safe_gcs_uri]
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print("Upload successful!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Upload failed: {e}")
        print(f"Error: {e.stderr}")
        return False
    
def download_from_gcs(source_path, local_destination):
    """Download file from GCS bucket using gsutil command line via Python"""
    gcs_prefix = f"{source_path}"
    result = subprocess.run(
                            ["gsutil", "ls", gcs_prefix],
                            check=True,
                            capture_output=True,
                            text=True
                        )
    folders = result.stdout.strip().split("\n")
    latest_folder = sorted(folders)[-1]   # assumes lexicographic order = time order

    # build the full path to predictions.jsonl
    gcs_uri = f"{latest_folder}predictions.jsonl"

    print(f"Downloading from {gcs_uri}")

    safe_gcs_uri = shlex.quote(gcs_uri)
    safe_local_path = shlex.quote(local_destination)
    
    command = f"gsutil cp {safe_gcs_uri} {safe_local_path}"
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print(f"Download successful!")
        print(f"Downloaded: {safe_gcs_uri} → {safe_local_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        print(f"Error: {e.stderr}")
        return False

def quick_response_check(jsonl_file):
    with open(jsonl_file, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())
            response = data.get('response', {})
            candidates = response.get('candidates', [])
            
            if candidates and candidates[0].get('content', {}).get('parts'):
                text = candidates[0]['content']['parts'][0].get('text', '')
                print(f"Line {i+1}: {text[:50]}...")
            else:
                print(f"Line {i+1}: No valid response")
            
            if i >= 4:  # Show first 5 only
                break

def read_jsonl_optimized(df, jsonl_file, location_guidance=False):
    """Read the JSONL file and update the DataFrame with the predictions"""
    updates = []
    if not location_guidance:
        new_col_name = f'gemini-{version}_faiss_loc'
    else:
        new_col_name = f'gemini-{version}_faiss_country'
    with open(jsonl_file, 'r') as f:
        for i, line in enumerate(f):
            if i % 10000 == 0:
                print('Processed line number', i)

            try:
                data = json.loads(line.strip())
                img_id = int(data.get('key', None))
                answer_text = 'no'
                candidates = data["response"]["candidates"]
                for candidate in candidates:
                    if "parts" in candidate["content"]:
                        for part in candidate["content"]["parts"]:
                            # We only care about the final answer text
                            if not part.get("thought", False):
                                answer_text = part["text"]
                                break  # Found the answer, move to the next candidate/line
                
                # Append the necessary data to the list
                updates.append({
                    'id_new': img_id,
                    new_col_name: str(answer_text).lower()
                })
                
            except Exception as e:
                # For failed records, we still need to record the 'no' status if not already set
                updates.append({
                    'id_new': img_id,
                    new_col_name: 'no'
                })
                continue
                
    # 3. Vectorized DataFrame Update (the fast part)
    if updates:
        # Convert the list of updates to a temporary DataFrame
        updates_df = pd.DataFrame(updates)
        print('***************', len(updates_df), updates_df.id_new.head(5), df.id_new.head(5))
        # Merge the updates into the main DataFrame in one go
        # This is millions of times faster than using .loc in a loop
        df = df.merge(
            updates_df, 
            on='id_new', 
            how='left', 
            suffixes=('', '_new')
        )
        # Coalesce the columns: use the new value if available, otherwise keep the old
        new_col_name = f'gemini-{version}_faiss_loc'
        new_col_name_temp = f'{new_col_name}_new'
        
        # This line performs the update for all 700K rows in a single operation
        # df[new_col_name] = df[new_col_name_temp].fillna(df[new_col_name])
        # df = df.drop(columns=[new_col_name_temp])
        
    return df
            
def read_jsonl(df, jsonl_file):
    """Read the JSONL file and update the DataFrame with the predictions"""
    with open(jsonl_file, 'r') as f:
        for i, line in enumerate(f):
            print('processed line number', i)
            data = json.loads(line.strip())
            img_id = data.get('key', None)
            try:
                candidates = data["response"]["candidates"]
            except Exception as e:
                print("Skipping for id", img_id)
                df.loc[df['id_new'] == img_id, f'gemini-{version}_faiss_loc'] = 'no'
                continue
            thoughts_text = ""
            answer_text = ""
            for candidate in candidates:
                if "parts" not in candidate["content"]:
                    answer_text = "no"
                    thoughts_text = "no"
                    continue
                parts = candidate["content"]["parts"]
                for part in parts:
                    if part.get("thought", False):  # This is thinking text
                        thoughts_text = part["text"]
                    else:  # This is the actual response
                        answer_text = part["text"]
            thoughts_text = ' '.join(thoughts_text.splitlines())
            thoughts_text = ' '.join(thoughts_text.split())

            
            df.loc[df['id_new'] == img_id, f'gemini-{version}_faiss_loc'] = str(answer_text).lower()
            
    return df


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')



def create_jsonl_file(df, csv_path, thinking_budget=-1, text_col='text', location_guidance=False):
    from google.genai import types
    from google import genai
    
    filename = os.path.basename(csv_path)
    filename = filename.replace(".csv", ".jsonl")
    folder = os.path.dirname(csv_path)
    with open("geonames_obscure_places.json", "r") as f:
        geo_db = json.load(f)
    for key in geo_db:
        geo_db[key] = [x['country'] for x in geo_db[key]]
    
    """
    0-shot prompt
    prompt = f"You are a geotagging agent who tags each given text to a country, if a reference to a location is present in the text. The only output you give is either the coutry name or 'NO' in case the text cannot be tagged to a country. Do NOT use abbreviations for country names. 

    Here are a few instructions. 
    Read the text carefully and understand the context.
    Find any location specified in the text, and then try to map it to a country name. If no location is specified, output 'no'.
    Do NOT use abbreviations for any country name. 
    Do NOT just mention the inferred location name. Only mention the mapped country name, and nothing else.
    You cannot mention the US state name if present. You should predict United States for them.
    All countries and places under United Kingdom (e.g., england, london, ireland, scotland, wales, etc) should be marked as United Kingdom.
    All states, cities or places under United States should be marked as United States.
    Do NOT use terms like America, USA, United States of America, for United States.
    Do NOT make any assumptions about a caption for which the country name cannot be inferred directly from the caption.
    Do NOT make any assumptions about a caption for which no location is directly specified in the caption.
    Output should be no if only the continent name is specified, or other vague regions are mentioned in the caption like 'mediterranean', 'caribbean', etc.
    Ensure that the output only contains the country name, or no, if no country can be inferred from the caption.
    "
    """
    """Country extraction prompt with ICL
    prompt = f"You are a geotagging agent who tags each given text to a country, if a reference to a location is present in the text. The only output you give is either the coutry name or 'NO' in case the text cannot be tagged to a country. Do NOT use abbreviations for country names. 

    Here are a few example texts with their corresponding geotags as examples.
    {example}

    Here are a few instructions. 
    Read the text carefully and understand the context.
    Find any location specified in the text, and then try to map it to a country name. If no location is specified, output 'no'.
    Do NOT use abbreviations for any country name. 
    Do NOT just mention the inferred location name. Only mention the mapped country name, and nothing else.
    You cannot mention the US state name if present. You should predict United States for them.
    All countries and places under United Kingdom (e.g., england, london, ireland, scotland, wales, etc) should be marked as United Kingdom.
    All states, cities or places under United States should be marked as United States.
    Do NOT use terms like America, USA, United States of America, for United States.
    Do NOT make any assumptions about a caption for which the country name cannot be inferred directly from the caption.
    Do NOT make any assumptions about a caption for which no location is directly specified in the caption.
    Output should be no if only the continent name is specified, or other vague regions are mentioned in the caption like 'mediterranean', 'caribbean', etc.
    Ensure that the output only contains the country name, or no, if no country can be inferred from the caption.
    "
    """
    if not location_guidance:
        #Location extraction prompt
        prompt = """You are a geoparsing agent who extracts the primary location mentioned in a text, if a reference to a location is present in the text. The only output you give is either the location name or 'NO' in case the text cannot be tagged to a country. Do NOT use abbreviations for country names. 

        Here are a few instructions. 
        Read the text carefully and understand the context.
        Only output the location name, as present in the text. Do not convert it to a country name.
        Only in case the country name is directly mentioned in the text, output the country name.
        Find any location specified in the text. If no location is specified, output 'no'.
        Do NOT use abbreviations for any country name or location name. 
        If the location is United Kingdom, the output should be United Kingdom, not any other abbreviation like UK, U.K., GB, Great Britain, etc.
        If the location is United States, the output should be United States, not any other abbreviation like USA, America, etc.
        Do NOT use terms like America, USA, United States of America, for United States.
        Do NOT make any assumptions about a text for which the location name cannot be inferred directly from the text.
        Do NOT make any assumptions about a text for which no location is directly specified in the text.
        Output should be no if only the continent name is specified, or other vague regions are mentioned in the text like 'mediterranean', 'caribbean', etc.
        Ensure that the output only contains the location name, or no, if no location can be inferred from the text.
        """
    else:
        prompt = None
    with open(folder+'/'+filename, "w") as f:
        for _, row in df.iterrows():
            if location_guidance:
                list_of_places = ast.literal_eval(row['gemini-flash_faiss'])
            
                examples = ""
                if list_of_places == ['no']:
                    examples = ""
                else:
                    for place in list_of_places:
                        if place in geo_db:
                            countries = " ".join(geo_db[place])
                            examples += f"Location: {place}, geotag: {countries}.\n\n"
                # location_guidance
                prompt = f"""You are a geotagging agent who tags each given text to a country, if a reference to a location is present in the text. The only output you give is either the coutry name or 'NO' in case the text cannot be tagged to a country. Do NOT use abbreviations for country names. 
                
                Here are a few examples of locations and their corresponding geotags. You can use this to identify the location and the country name. If the detected location is not present in the examples, you can use your own knowledge to identify the country name.
                {examples}

                Here are a few instructions. 
                Read the text carefully and understand the context.
                Find any location specified in the text, and then try to map it to a country name. If no location is specified, output 'no'.
                Do NOT use abbreviations for any country name. 
                Do NOT just mention the inferred location name. Only mention the mapped country name, and nothing else.
                You cannot mention the US state name if present. You should predict United States for them.
                All countries and places under United Kingdom (e.g., england, london, ireland, scotland, wales, etc) should be marked as United Kingdom.
                All states, cities or places under United States should be marked as United States.
                Do NOT use terms like America, USA, United States of America, for United States.
                Do NOT make any assumptions about a caption for which the country name cannot be inferred directly from the caption.
                Do NOT make any assumptions about a caption for which no location is directly specified in the caption.
                Output should be no if only the continent name is specified, or other vague regions are mentioned in the caption like 'mediterranean', 'caribbean', etc.
                Ensure that the output only contains the country name, or no, if no country can be inferred from the caption.
                """
            img_id = row['id_new']
            text = row[text_col]
            
            # image_file = client.files.upload(file=row['local_imagepath']) #standalone genai sdk
            request_obj = {
                # "key": f"{row['custom_id']}", 
                "key": f"{img_id}",                    
                "request": {
                            "contents": [{
                                "role": "user",
                                "parts": [
                                        {"text": prompt + "Here is the text: " + str(text) + "### Final Answer:"},
                                    ]
                                }],
                                "generation_config": {
                                
                                    "thinkingConfig": {
                                                        "includeThoughts": True,
                                                        "thinkingBudget": 128
                                                    }
                                    }
                            }
                            }
            f.write(json.dumps(request_obj) + "\n")

    return folder+'/'+filename


call = "multiple"

# Parse command line arguments


newcsvpath = args.newcsvpath
text_col = args.text_col
df = pd.read_csv(newcsvpath)
if 'id_new' not in df.columns:
    df['id_new'] = [i for i in range(len(df))]
# df['id'] = df['id'].astype(str)
thinking_budget = 0
jsonl_path = create_jsonl_file(df, newcsvpath, thinking_budget, text_col, args.location_guidance)
filename = os.path.basename(jsonl_path)
upload = upload_with_subprocess(jsonl_path, "geoprofiler162", "")
from google import genai
from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions

client = genai.Client(vertexai=True,
                    project='geodiv4',
                    location="us-east5",
                    http_options=HttpOptions(api_version="v1"))
output_uri = f"gs://geoprofiler162/captions_results"


job = client.batches.create(
        # To use a tuned model, set the model param to your tuned model using the following format:
        # model="projects/{PROJECT_ID}/locations/{LOCATION}/models/{MODEL_ID}
        model=model,
        # Source link: https://storage.cloud.google.com/cloud-samples-data/batch/prompt_for_batch_gemini_predict.jsonl
        src=f"gs://geoprofiler162/{filename}",
        config=CreateBatchJobConfig(dest=output_uri),
    )

completed_states = {
    JobState.JOB_STATE_SUCCEEDED,
    JobState.JOB_STATE_FAILED,
    JobState.JOB_STATE_CANCELLED,
    JobState.JOB_STATE_PAUSED,
}

print(f"Monitoring job: {job.name}")
start_time = datetime.now()
check_interval = 100 #seconds
while True:
    try:
        job = client.batches.get(name=job.name)
        elapsed = datetime.now() - start_time
        
        print(f"[{elapsed}] Job state: {job.state}")
        
        if job.state in completed_states:
            print(f"Job completed with state: {job.state}")
            break
            
        time.sleep(check_interval) 
        
    except Exception as e:
        print(f"Error checking job status: {e}")
        time.sleep(check_interval)
        continue


if job.state.name == 'JOB_STATE_FAILED':
    if job.error:
        print(f"Error: {job.error}")
    else:
        print("Job failed but no error details available")
        
    
if job.state == "JOB_STATE_SUCCEEDED":
    print('Job succeeded')
    download_from_gcs(output_uri, 'data'+f'/captions_results.jsonl')
    # print()
    # print('Checking if the saved file has response field:')
    # quick_response_check(results_dir+f'/{concept_path}_predictions.jsonl')
    
    # 
    df = read_jsonl_optimized(df, 'data/captions_results.jsonl', args.location_guidance)
    df.to_csv(newcsvpath, index=False)
    os.remove('data'+f'/captions_results.jsonl')
    os.remove(f'{newcsvpath.replace(".csv", ".jsonl")}')
    





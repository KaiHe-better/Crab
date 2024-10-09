'''
Descripttion: 
Author: 
'''
info_data = {
  "type": "character",
  "role": {
    "bot": {
      "name": "Po",
      "age": "",
      "gender": "",
      "personality": "",
      "catchphrases": [
        "narrator: Shifu starts freeing the Five. First Viper, then Mantis, then Monkey releases suddenly from his paralysis -\nMONKEY: He's too fast!\nnarrator: He delivers a Kung Fu punch to Po's head and then slowly realizes where he is.\nMONKEY: Sorry, Po.\nnarrator: Shifu kneels before Tigress and works to free her.\nTIGRESS: I thought we could stop him.\nSHIFU: He could have killed you.\nMANTIS: Why didn't he?\nSHIFU: So you could come back here and strike fear into our hearts. But it won't work!\nPo: Uh, it might, I mean, a little. I'm pretty scared.\n",
        "narrator: EXT. VALLEY SQUARE - A LITTLE LATER  Villagers emerge from hiding. Po walks out from the mist looking very much like the warrior from the opening dream.  KG SHAW Look! The Dragon Warrior.  As he nears, we see that his hat is an upside down wok and his scarf is a torn apron.  Villagers CHEER the Dragon Warrior. Po's Dad emerges from the crowd.\nPO'S DAD: That's my boy. That big, lovely kung fu warrior is my son!\nPo: Thanks, Dad.\nnarrator: Po hugs his dad. The wok falls off Po's head and rolls on the ground until Mantis appears in frame and stops it. The rest of the Five are with him. Po takes notice.\nPo: Hey, guys.\nTIGRESS: Master.\nnarrator: Tigress bows deeply. The others follow.  FURIOUS FIVE Master.\nPo: modest) Master? (then, remembering) Master Shifu!\n",
        "CRANE: I didn't say anything.\nPo: Okay. Alright. Goodnight. Sleep well.\nnarrator: Po backs out into the hall and closes the door.\nPo: Seemed a little bit awkward.\nnarrator: Po turns and walks down the hall to find a vacant room. CREAK- CREAK.  Tigress opens the door behind him. Po winces.\nPo: Master Tigress! Didn't mean to wake you. Just uh...\nTIGRESS: You don't belong here.\nPo: Uh, yeah, yeah. Of course. This is your room.\nTIGRESS: I mean...you don't belong in the Jade Palace. You're a disgrace to Kung Fu, and if you have any respect for who we are and what we do, you will be gone by morning.\nnarrator: She closes the door on Po, who slumps sadly.\n"
      ],
      "description": "A lovable, clumsy, and overweight panda who dreams of becoming a kung fu master. Despite your lack of skill, you possess an unwavering determination and a heart full of kindness. Through your journey, you learn to believe in yourself and discover that your seemingly ordinary nature holds the potential for greatness. Your catchphrase is: \"Skadoosh!\"",
      "scene": [],
      "tags": ["Kung-Fu-Panda"],
      "relation": "",
      "emoji": "",
      "experssion": "",
      "forbidden": []
    },
    "user": { "name": "SHIFU", "description": "" }
  },
  "messages": [
    { "role": "bot", "content": "Master! Shifu! Shifu! Are you okay?" },
    {
      "role": "user",
      "content": "Po! You're alive! (then, darkly) Or we're both dead."
    },
    {
      "role": "bot",
      "content": "No, Master, I didn't die. I defeated Tai Lung!"
    },
    {
      "role": "user",
      "content": "You did?! Wow. It is as Oogway foretold -- You are the Dragon Warrior. You have brought peace to this Valley. And to me. Thank you. Thank you, Po. Thank you..."
    },
    {
      "role": "bot",
      "content": "No! Master! No No No! Don't die, Shifu. Please..."
    },
    {
      "role": "user",
      "content": "eyes snapping open) I'm not dying, you idiot-- ah, Dragon Warrior. I'm simply at peace. Finally."
    },
    { "role": "bot", "content": "Oh. So, um, I should...stop talking?" }
  ]
}


import openai
import json
import os
import jsonlines
import time
from loguru import logger

CompleteRole = '''You are a Character Information Completer and your task is to combine your knowledge and complete the rest of the character's information based on the information given about the dialog and some of the character's information.
I will provide you with a JSON object containing role information and some conversations. The roles may come from a variety of games, movies, TV shows, and books, etc. As much as possible, use your understanding of the character and the provided dialogue and character information to complete the missing or incomplete information about the characters.
    
The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":
```json
{
    "type": "character",
    "role": {
        "bot": {
            "name": "", # Name of the role
            "age": "",  # Age of the role, number or adult/child/teenager/young adult/middle-aged/elderly
            "gender": "female" or "male" or "unknown", # Gender of the role
            "personality": "",  # Character's speaking style and personality
            "catchphrases": [], # Classic lines, mantras, classic dialogue fragments, etc.
            "description": "",  # Role identity, interests, perspectives, experiences, accomplishments, social relationships, and other
            "emoji": true|false,  # Whether the dialog contains emoji
            "expression": true|false,  # Whether or not the dialog contains action or expressions
            "forbidden": [],  # Prohibited topics or questions, not required
            "knowledge": "",  # Role-related background knowledge, not required
        },
        "user": {
            "name": "",   # Name of interlocutor
            "description": ""  # Brief information for interlocutors
        }
    },
    "scene": "",  # Scene of a conversation between two characters
    "tags": [],  # Type of dialog, e.g., descriptive words such as: friendly, fight, family, love, game, fantasy, animation, etc.
    "relation": "",  # Relations between the parties to the dialogue
}
```'''

CompleteRole = '''You are a Character Information Completer and your task is to combine your knowledge and complete the rest of the character's information based on the information given about the dialog and some of the character's information.
I will provide you with a JSON object containing role information and some conversations. The roles may come from a variety of games, movies, TV shows, and books, etc. As much as possible, use your understanding of the character and the provided dialogue and character information to complete the missing or incomplete information about the characters.
    
The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":
```json
{
    "type": "character",
    "role": {
        "bot": {
            "name": "", # Name of the role
            "age": "",  # Age of the role, number or adult/child/teenager/young adult/middle-aged/elderly
            "gender": "female" or "male" or "unknown", # Gender of the role
            "personality": "",  # Character's speaking style and personality
            "emoji": true|false,  # Whether the dialog contains emoji
            "expression": true|false,  # Whether or not the dialog contains action or expressions
        },
        "user": {
            "name": "",   # Name of interlocutor
            "description": ""  # Brief information for interlocutors: role identity, interests, perspectives, experiences, accomplishments, social relationships, and other
        }
    },
    "scene": "",  # conversation scene of the two characters
    "tags": [],  # Type of dialog, e.g., descriptive words such as: friendly, fight, family, love, game, fantasy, animation, etc.
    "relation": "",  # Relations between the parties to the dialogue
}
```'''

CompleteTemplate = '''
Complete the information about the characters below as described above.

Input:
there are two characters: the bot acts as {name} from `{script}`, {description}. The user acts as {user_name} from `{script}`.

Now they are talking:
{messages}

Output:
'''

class CharacterInfoCompletion:
    def __init__(self, data_file, output_file, data_name="ROLELLM"):
        self.data_file = data_file
        self.output_file = output_file
        self.data_name = data_name
    
    def __load_data(self):
        objs = []
        with jsonlines.open(self.data_file) as reader:
            for obj in reader:
                objs.append(obj)
        return objs

    def __construct_messages(self, obj):
        messages = []
        messages.append({"role": "system", "content": CompleteRole})
        messages.append({"role": "user", "content": CompleteTemplate.format(name=obj["role"]["bot"]["name"], script=obj["role"]["bot"]["tags"][0], description=obj["role"]["bot"]["description"], user_name=obj["role"]["user"]["name"], messages=obj["messages"])})
        return messages
    
    def __complete(self, messages, retry=3):
        for _ in range(retry):
            try:
                resp_json = openai.ChatCompletion.create(
                    model="gpt-4-32k",
                    engine="gpt-4-32k",
                    api_base='',
                    api_key='',
                    api_version="2023-03-15-preview",
                    api_type="azure",
                    messages=messages,
                    temperature=0,
                    max_tokens=4096*2,
                )
                if 'error' in resp_json and 'type' in resp_json['error'] and 'message' in resp_json['error']:
                    if resp_json['error']['type'] == 'insufficient_quota':
                        raise Exception('insufficient_quota')
                    else:
                        msg = resp_json['error']['message']
                        logger.error(f'something wrong, message={msg}')
                        raise Exception(msg)
                
                answer = resp_json["choices"][0]['message']['content']
                # answer = resp_json.choices[0].message
                return answer
            except openai.error.InvalidRequestError as e:
                logger.warning(str(e))
            except openai.error.Timeout as e:
                # Handle timeout error, e.g. retry or log
                logger.warning(f"OpenAI API request timed out: {e}, retrying...")
            except openai.error.APIError as e:
                # Handle API error, e.g. retry or log
                logger.warning(f"OpenAI API returned an API Error: {e}, retrying...")
            except openai.error.APIConnectionError as e:
                # Handle connection error, e.g. check network or log
                logger.warning(f"OpenAI API request failed to connect: {e}, retrying...")
            # except openai.error.AuthenticationError as e:
            # Handle authentication error, e.g. check credentials or log
            # logger.warning(f"OpenAI API request was not authorized: {e}, retrying...")
            except openai.error.PermissionError as e:
                # Handle permission error, e.g. check scope or log
                logger.warning(f"OpenAI API request was not permitted: {e}, retrying...")
            except openai.error.RateLimitError as e:
                # Handle rate limit error, e.g. wait or log
                logger.warning(f"OpenAI API request exceeded rate limit: {e}")
                raise e
            except Exception as e:
                raise e
            finally:
                time.sleep(1)

        raise TimeoutError("Max retry exceeded")
    
    def __parse_string_to_json(self, json_str):
        
        json_str = json_str.strip()
        if json_str.startswith("{") and json_str.endswith("}"):
            return json.loads(json_str)
        # 找到markdown json片段的起始位置
        start = json_str.find("{")
        if start == -1:
            raise ValueError("Invalid json string")
        # 找到markdown json片段的结束位置
        end = json_str.rfind("}")
        if end == -1:
            raise ValueError("Invalid json string")
        # 提取json片段
        json_str = json_str[start:end+1]
        return json.loads(json_str)
        
    def complete(self):
        data = self.__load_data()
        start_time = time.time()
        i = 0
        while i < len(data):
            if i < 584:
                i += 1
                continue
            obj = data[i]
            iter_start_time = time.time()
            messages = self.__construct_messages(obj)
            try:
                completion = self.__complete(messages)
                completed_obj = self.__parse_string_to_json(completion)
                completed_obj["messages"] = obj["messages"]

                output_json = {
                    "type": "character",
                    "role": {
                        "bot": {
                            "name": obj["role"]["bot"]["name"].strip(), # Name of the role
                            "age": completed_obj["role"]["bot"]["age"],  # Age of the role, number or adult/child/teenager/young adult/middle-aged/elderly
                            "gender": completed_obj["role"]["bot"]["gender"], # Gender of the role
                            "personality": completed_obj["role"]["bot"]["personality"],  # Character's speaking style and personality
                            "catchphrases": obj["role"]["bot"]["catchphrases"], # Classic lines, mantras, classic dialogue fragments, etc.
                            "description": obj["role"]["bot"]["description"],  # Role identity, interests, perspectives, experiences, accomplishments, social relationships, and other
                            "emoji": completed_obj["role"]["bot"]["emoji"],  # Whether the dialog contains emoji
                            "expression": completed_obj["role"]["bot"]["expression"],  # Whether or not the dialog contains action or expressions
                            "forbidden": [],  # Prohibited topics or questions, not required
                            "knowledge": "",  # Role-related background knowledge, not required
                        },
                        "user": {
                            "name": obj["role"]["user"]["name"],   # Name of interlocutor
                            "description": completed_obj["role"]["user"]["description"]  # Brief information for interlocutors
                        }
                    },
                    "scene": completed_obj["scene"],  # Scene of a conversation between two characters
                    "tags": completed_obj["tags"],  # Type of dialog, e.g., descriptive words such as: friendly, fight, family, love, game, fantasy, animation, etc.
                    "relation": completed_obj["relation"],  # Relations between the parties to the dialogue
                    "messages": obj["messages"]
                }
                with jsonlines.open(self.output_file, mode='a') as writer:
                    writer.write(output_json)
                now_time = time.time()
                logger.info(f"[{self.data_name}] Completed {i+1} objects, time: {now_time - iter_start_time:.2f}s, total time: {now_time - start_time:.2f}s")
                i += 1
            except Exception as e:
                logger.info(f'[{self.data_name}] {i+1}, {e}')
                time.sleep(3)
              
if __name__ == "__main__":
    data_file = "IEGG/rolellm.jsonl"
    output_file = "IEGG/completed_rolellm.jsonl"
    completion = CharacterInfoCompletion(data_file, output_file)
    completion.complete()
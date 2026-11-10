from  pydantic import BaseModel ,EmailStr ,Field 
from fastapi import FastAPI , Response ,status , HTTPException ,Depends , APIRouter,Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from app.celery_app import celery
import json
from typing import Literal
from pathlib import Path

BASE=Path(__file__).resolve().parent
CONV_FILE=BASE / "conversations_dataset.jsonl"
ANALYZED_FILE=BASE/ "message_analyzed_dataset.jsonl"

OUT_DIR = Path("D:/synthetic_output")          # outside the project, so Live Server won't refresh
OUT_DIR.mkdir(exist_ok=True)

FILES = {
    "english": {
        "conv":     "D:/data_traning/new_traning_data/english_conversations.jsonl",
        "ana":      "D:/data_traning/new_traning_data/english_analyzed.jsonl",
        "out_conv": OUT_DIR / "clean_english_conversations.jsonl",
        "out_ana":  OUT_DIR / "clean_english_analyed.jsonl",
    },

    "urdu": {
        "conv":     "D:/data_traning/new_traning_data/urdu_conversations.jsonl",
        "ana":      "D:/data_traning/new_traning_data/urdu_analyzed.jsonl",
        "out_conv": OUT_DIR / "clean_urdu_conversations.jsonl",
        "out_ana":  OUT_DIR / "clean_urdu_analyed.jsonl",
    },
}


app=FastAPI()

origins=["https://www.google.com","http://127.0.0.1:5500"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



OLLAMA_BASE_URL = "http://localhost:11434/v1"

ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key='ollama')

status=""

system_prompt1="""
- you are a conversation generator you will generate conversation either in single mode or in group mode 

- always know Me wll be the USER and others can be those who are chatting with him

- you will generate in json format the data

- you will generate onversations in english

- some examples of data are below

- dont use emojis only use text in the messages

example 1
 
{
  "Model_tone": "casual",
  "Mode": "single",
  "conversation": [
    {
      "speaker": "Sara",
      "relationship": "friend",
      "text": "Are you free tonight?"
    },
    {
      "speaker": "Me",
      "relationship": "Me",
      "text": "Why? What's up?"
    },
    {
      "speaker": "Sara",
      "relationship": "friend",
      "text": "Do you want to go out and grab something to eat?"
    }
  ]
}
  

 example 2

 {
  "Model_tone": "supportive",
  "Mode": "group",
  "conversation": [
    {
      "speaker": "Ayesha",
      "relationship": "friend",
      "text": "Guys, we have a presentation tomorrow. Is everyone ready?"
    },
    {
      "speaker": "Usman",
      "relationship": "colleague",
      "text": "My slides are complete. Only Talha's part is left."
    },
    {
      "speaker": "Hira",
      "relationship": "friend",
      "text": "Talha, when are you going to send your part? I've been messaging you since morning."
    },
    {
      "speaker": "Talha",
      "relationship": "Bestfriend",
      "text": "Sorry guys, we had guests over at home. I'll start working on it now."
    },
    {
      "speaker": "Ayesha",
      "relationship": "friend",
      "text": "Starting now? The presentation is tomorrow at 9 AM!"
    },
    {
      "speaker": "Usman",
      "relationship": "colleague",
      "text": "I knew he would leave it until the last minute."
    },
    {
      "speaker": "Talha",
      "relationship": "Bestfriend",
      "text": "Hey, why are you getting so angry? I'll send it by tonight."
    },
    {
      "speaker": "Hira",
      "relationship": "friend",
      "text": "By tonight, you mean midnight? Then who's going to fix the design?"
    },
    {
      "speaker": "Me",
      "relationship": "Me",
      "text": "Talha, how many slides do you still have left?"
    },
    {
      "speaker": "Talha",
      "relationship": "Bestfriend",
      "text": "Just four slides, and I still have to do the research."
    },
    {
      "speaker": "Ayesha",
      "relationship": "friend",
      "text": "@Me, could you help Talha out? You've already finished your part."
    },
    {
      "speaker": "Usman",
      "relationship": "colleague",
      "text": "Yeah, please. Otherwise, all of us will end up getting bad grades."
    },
    {
      "speaker": "Hira",
      "relationship": "friend",
      "text": "Please, Me. We're all counting on you."
    }
  ]
}

 example 3:

 {
  "Model_tone": "de-escalating",
  "Mode": "single",
  "conversation": [
    {
      "speaker": "Bilal",
      "relationship": "Bestfriend",
      "text": "Dude, where are you? I've been waiting for an hour."
    },
    {
      "speaker": "Me",
      "relationship": "Me",
      "text": "Sorry, man. I got stuck in traffic."
    },
    {
      "speaker": "Bilal",
      "relationship": "Bestfriend",
      "text": "You said 6 PM. It's already 7!"
    },
    {
      "speaker": "Me",
      "relationship": "Me",
      "text": "I'll be there in just 10 minutes."
    },
    {
      "speaker": "Bilal",
      "relationship": "Bestfriend",
      "text": "You always come up with the same excuse."
    },
    {
      "speaker": "Me",
      "relationship": "Me",
      "text": "I'm genuinely telling the truth this time. The traffic was really bad."
    },
    {
      "speaker": "Bilal",
      "relationship": "Bestfriend",
      "text": "I don't care. You've wasted my entire evening."
    },
    {
      "speaker": "Bilal",
      "relationship": "Bestfriend",
      "text": "I'm not making plans with you again."
    }
  ]
}
 
- generate in same format as in above examples

"""

system_prompt2="""
you are a chat response generator you will be give some conversation and need to generate the next message the User me can send based on the conversations

You must determine:


1. language:
   The language should be english
  


2. type:
   - "single" if the conversation is between the user and one other person.
   - "group" if multiple people are participating.


3. messages:
   - Generate 3-4 possible replies that "Me" could send next.
   - Each reply must be short and natural.
   - Replies must fit the current context.
   - Do not repeat the same idea.
   - Use the same language/script as the conversation.
   - The replies should feel like something a real person would send in a chat.


4. conversation_tone:
   Determine the overall tone of the OTHER participants' messages.
   Choose one:
   - casual
   - formal
   - happy
   - sad
   - angry
   - neutral

5. conversation_topic:
   Summarize the main topic in a few words.

6. Me_tone:
   Determine the tone used by "Me" in their existing messages.
   Choose one:
   - casual
   - formal
   - funny
   - supportive
   - neutral
   - escalating
   - descalating


Important rules:
- Return ONLY valid JSON.
- Do not return markdown.
- Do not use ```json.
- Do not add explanations before or after the JSON.
- Follow the exact field names and allowed values from the ChatResponse schema


example 1

"user_prompt" : {

 {
 "Model_tone":"descalating"
 "Mode":"single",
  "conversation": [
    {
      "speaker": "Bilal",
      "relationship": "Bestfriend",
      "text": "Dude, where are you? I've been waiting for an hour."
    },
    {
      "speaker": "Me",
      "relationship": "Me",
      "text": "Sorry, man. I got stuck in traffic."
    },
    {
      "speaker": "Bilal",
      "relationship": "Bestfriend",
      "text": "You said 6 o'clock, and now it's already 7."
    },
    {
      "speaker": "Me",
      "relationship": "Me",
      "text": "I'll be there in just 10 minutes."
    },
    {
      "speaker": "Bilal",
      "relationship": "Bestfriend",
      "text": "You always come up with the same excuse."
    },
    {
      "speaker": "Me",
      "relationship": "Me",
      "text": "I really did get stuck in traffic this time. There was a massive traffic jam."
    },
    {
      "speaker": "Bilal",
      "relationship": "Bestfriend",
      "text": "I don't care. You've wasted my entire time."
    },
    {
      "speaker": "Bilal",
      "relationship": "Bestfriend",
      "text": "I'm not making plans with you again."
    }
  ]
 
 }

}


you will generate these things according to the conversations



{
  "language": 
  "type": 
  "messages": the message that user me should answer with accoridng to conversation (give 1 different messages options in each tone type positive negative neutral and supportive) in a list format and generate messages in roman urdu no other language not pure urdu roman urdu
  e.g{
  "tone":"Negative",
  "message":"get lost dont disturb me such a waste of energy"
  }

  "conversation_tone": "casual",
  "conversation_topic": "going out and time issue",
  "Me_tone": "neutral",
  "conversation_focus": "what is the conversation focus and what message to relpy"
}


these is an example

{
  "language": "english",
  "type": "single",
  "messages": [
    {
      "tone": "positive",
      "message": "Sorry, man. I wasted your time, and I understand that. I'll make sure to be on time next time."
    },
    {
      "tone": "negative",
      "message": "Come on, what's the big deal? It's not like I can control the traffic."
    },
    {
      "tone": "neutral",
      "message": "I know you've been waiting for an hour. I'll be there in just 10 minutes."
    },
    {
      "tone": "supportive",
      "message": "I understand why you're upset, and I'm sorry. It was my mistake. Please try to understand, it won't happen again."
    }
  ],
  "conversation_tone": "de-escalating",
  "conversation_topic": "late arrival and waiting",
  "Me_tone": "neutral",
  "conversation_focus": "Calming Bilal down, explaining the reason for arriving late, and resolving the tension between friends"
}

   
example 2

this is and example of a conversation 
 {   

 "Model_tone":"supportive",
 "Mode":"group",
  "conversation": [
    {
      "speaker": "Ayesha",
      "relationship": "friend",
      "text": "Guys, we have a presentation tomorrow. Is everyone ready?"
    },
    {
      "speaker": "Usman",
      "relationship": "colleague",
      "text": "My slides are complete. Only Talha's part is left."
    },
    {
      "speaker": "Hira",
      "relationship": "friend",
      "text": "Talha, when are you going to send your part? I've been messaging you since morning."
    },
    {
      "speaker": "Talha",
      "relationship": "Bestfriend",
      "text": "Sorry, guys. We had guests over at home. I'll start working on it now."
    },
    {
      "speaker": "Ayesha",
      "relationship": "friend",
      "text": "Starting now? The presentation is tomorrow at 9 AM!"
    },
    {
      "speaker": "Usman",
      "relationship": "colleague",
      "text": "I knew he would leave it until the last minute."
    },
    {
      "speaker": "Talha",
      "relationship": "Bestfriend",
      "text": "Hey, why are you getting so angry? I'll send it by tonight."
    },
    {
      "speaker": "Hira",
      "relationship": "friend",
      "text": "By tonight, you mean midnight? Who's going to fix the design then?"
    },
    {
      "speaker": "Me",
      "relationship": "Me",
      "text": "Talha, how many slides do you still have left?"
    },
    {
      "speaker": "Talha",
      "relationship": "Bestfriend",
      "text": "Just four slides, and I still have to do the research."
    },
    {
      "speaker": "Ayesha",
      "relationship": "friend",
      "text": "Me, could you help Talha out? You've already finished your part."
    },
    {
      "speaker": "Usman",
      "relationship": "colleague",
      "text": "Yes, please. Otherwise, we'll all end up getting bad grades."
    },
    {
      "speaker": "Hira",
      "relationship": "friend",
      "text": "Please, Me. We're all counting on you."
    }
  ]

 }    

you will generate these things according to the conversations

Important NOTE: the messages will be roman urdu  according to tone of conversation not in pure urdu writing

{
  "language": 
  "type": 
  "messages": the message that user me should answer with accoridng to conversation (give 1 different messages options in each tone type positive negative neutral and supportive) in a list format and generate messages in roman urdu no other language not pure urdu roman urdu
  e.g{
  "tone":"positive",
  "message":"Taha okay i will hep you in your presentation"
  }

  "conversation_tone": "casual",
  "conversation_topic": "presentation preparation",
  "Me_tone": "neutral",
  "conversation_focus": "what is the conversation focus right now"
}


these is an example

{
  "language": "english",
  "type": "group",
  "messages": [
    {
      "tone": "positive",
      "message": "Don't worry, guys. I'll help Talha out. We can work together and finish the presentation on time."
    },
    {
      "tone": "negative",
      "message": "Talha, you should have completed your part earlier. Now you're delaying everyone's work."
    },
    {
      "tone": "neutral",
      "message": "Talha, send me your four slides. I'll check how much help you need with the research and design."
    },
    {
      "tone": "supportive",
      "message": "Don't worry, Talha. I'll help you out. We can finish the slides and research together. Just share whatever you've completed so far."
    }
  ],
  "conversation_tone": "supportive",
  "conversation_topic": "presentation preparation and teamwork",
  "Me_tone": "supportive",
  "conversation_focus": "Helping Talha complete his presentation and supporting the group in getting everything ready before the deadline"
}


"""


def callAI(message):
 response = ollama.chat.completions.create(model="qwen3:14b", messages=message)
 return response.choices[0].message.content

def callAI2(message):
    response = ollama.chat.completions.create(model="qwen3:14b", messages=message)
    return response.choices[0].message.content 

 
class Turn(BaseModel):
     speaker:str=Field(description="Name of the speaker, e.g. 'Sara' or the user's label")
     relationship:str=Field(description="relationship of the speaker with me")
     text:str=Field(description="one chat message,written like a real text message") 

class DatasetExample(BaseModel):

    Mode:str
    Model_tone:str = Field(
        description="Tone used in the suggested replies"
    )
    conversation: list[Turn] = Field(min_length=3, max_length=20)

class Message(BaseModel):

     tone:str=Field(description="What is the tone of the message")
     message:str


class ChatResponse(BaseModel):

    language: Literal["roman_urdu", "english" , "roman urdu"] = Field(
        description="Language and script of the suggested replies")
    
    type:Literal["single","group"] = Field(
        description="Detect is conversation is single or in group")
    
    messages:list[Message]=Field(description="3-4 suggested reply options,each a short text message")
    
    conversation_tone:str
  

    conversation_topic:str=Field(
        description="Topic of the conversation in a few words"
    )

    Me_tone:str = Field(
        description="Tone used by the user in the replies"
    )

    conversation_focus:str = Field(
        description="What the other person mainly wants or is asking about"
    )
   

class LoadReq(BaseModel):
    id: int | None = None

@app.post("/load_messages")
def load_messages(req:LoadReq):

   if req.id==None:
       iid=0
       element1=""
       element2=""
       
       with open("urdu_conversations.jsonl", "r", encoding="utf-8") as file:
         for i, line in enumerate(file):
           if i == iid:
            element1 = json.loads(line)
            print(element1)
            break
           
       with open("urdu_analyzed.jsonl", "r", encoding="utf-8") as file:
                for i, line in enumerate(file):
                  if i == iid:
                   element2 = json.loads(line)
                   print(element2)
                   break 
       data={}

       data["id"]=iid
       data["conversations"]=element1
       data["analysis"]=element2

       json_string = json.dumps(data)
       return {"id": iid, "conversation": element1, "analysis": element2}

   else:
       iid=req.id+1
       element1=""
       element2=""
       
       with open("urdu_conversations.jsonl", "r", encoding="utf-8") as file:
         for i, line in enumerate(file):
           if i == iid:
            element1 = json.loads(line)
            print(element1)
            break
           
       with open("urdu_analyzed.jsonl", "r", encoding="utf-8") as file:
                for i, line in enumerate(file):
                  if i == iid:
                   element2 = json.loads(line)
                   print(element2)
                   break 
       data={}

       data["id"]=iid
       data["conversations"]=element1
       data["analysis"]=element2

       json_string = json.dumps(data)
       return  {"id": iid, "conversation": element1, "analysis": element2}
        

class SendReq(BaseModel):
    id: int | None = None
    conversation: dict
    analysis: dict


# @app.post("/send_messages")
# def send_message(data_pydantic:SendReq):

#      data=data_pydantic.model_dump()
#      element1=data["conversation"]
#      element2=data["analysis"]


#      if element2["language"]=="english":
#       with open("new_english_analyed_dataset.jsonl", "a", encoding="utf-8") as f:
#         f.write(json.dumps(element2, ensure_ascii=False) + "\n") 
#         print("saved to analyze file")
#       with open("new_english_conversations_dataset.jsonl", "a", encoding="utf-8") as f:
#               f.write(json.dumps(element1, ensure_ascii=False) + "\n") 
#               print("saved to message file")

#      else:
#       with open("new_urdu_analyed_dataset.jsonl", "a", encoding="utf-8") as f:
#                 f.write(json.dumps(element2, ensure_ascii=False) + "\n") 
#                 print("saved to analyze file")
#       with open("new_urdu_conversations_dataset.jsonl", "a", encoding="utf-8") as f:
#                       f.write(json.dumps(element1, ensure_ascii=False) + "\n") 
#                       print("saved to message file")  


@app.post("/send_messages")
def send_message(data_pydantic: SendReq):

    data = data_pydantic.model_dump()
    element1 = data["conversation"]
    element2 = data["analysis"]

    lang = "english" if element2.get("language", "").strip().lower() == "english" else "urdu"
    f = FILES[lang]

    with open(f["out_ana"], "a", encoding="utf-8") as file:
        file.write(json.dumps(element2, ensure_ascii=False) + "\n")
        print("saved to analyze file:", f["out_ana"])

    with open(f["out_conv"], "a", encoding="utf-8") as file:
        file.write(json.dumps(element1, ensure_ascii=False) + "\n")
        print("saved to message file:", f["out_conv"])

    return {"ok": True}


def start_generating():
         
  try:     
         print("calling to the ai ")
         if status!="stop":

               user_prompt="""now generate a conversation on the format you are given"""
               
               message = [
    {"role": "system", "content": system_prompt1},
    {"role": "user", "content": user_prompt}
    ]
               result=callAI(message)
               print("========== AI RESPONSE ==========")
               print(type(result))
               print(result)
               print("=================================")    
               results = json.loads(result) 
               
               
               Response= DatasetExample.parse_obj(results)  #checking that conversation match
               if Response:
                       send=Response.model_dump()

                       send_string = json.dumps(send , ensure_ascii=False) # this we are making it like string

                       user_prompt2="""generate a chat response on these messages\n\n""" + send_string
                       message2 = [
                           {"role": "system", "content": system_prompt2},
                           {"role": "user", "content": user_prompt2}
                           ]
                       
                      #  result2=callAI2(message2)
                      #  print(result2)
                      #  results2 = json.loads(result2)

                      #  Response= ChatResponse.parse_obj(results)  #checking that conversation match
                       chat = None
                       for attempt in range(3):
                         result2=callAI2(message2)
                         text = (result2 or "").strip()
                         print("SECOND AI RAW:", repr(text[:500]))
                         if not text:
                           print(f"attempt {attempt + 1}: empty response")
                           continue
                         try:
                            chat= ChatResponse.model_validate_json(text)
                            break
                         except Exception as e:
                                print(f"attempt {attempt + 1}:validation failed:", e)
                       
                       print("giving up after 3 attempts")     

                       
                       if chat:

                                    with open(CONV_FILE, "a", encoding="utf-8") as f:
                                                                 f.write(json.dumps(send, ensure_ascii=False) + "\n") 
                                                                 print("saved to", CONV_FILE)
                                    
                                    with open("message_Analyzeed_dataset.jsonl", "a", encoding="utf-8") as f:
                                         f.write(json.dumps(chat.model_dump(), ensure_ascii=False) + "\n") 
                                    print("saved to", ANALYZED_FILE)


  except json.JSONDecodeError:
       print(" AI returned invalid json")

  except Exception as e:
       print("Generation failed:",e)     

@app.post("/stop")
async def stop_generating():

    global status
    status = "stop"
    return {"status": "stopped"}



@celery.task
def generate_order():

    start_generating()


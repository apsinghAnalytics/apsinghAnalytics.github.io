---
layout: post
title: VentureGen- A Startup Name and Pitch Generator App powered by LangChain and GPT 3.5  
image: VentureGenBackground.png
date: 2023-12-23 8:26:20 +0400
tags: [OpenAI, LangChain, Large Language Models (LLM), Streamlit, App, Natural Language Processing (NLP), GPT 3.5, LLM App Framework, Cloud]
categories: [App, LLM, Businesses]
---
Checkout the deployed app in action: [http://ec2-40-176-10-24.ca-west-1.compute.amazonaws.com:8501](http://ec2-40-176-10-24.ca-west-1.compute.amazonaws.com:8501) 

The link to the github repo for this app is [here](https://github.com/apsinghAnalytics/streamlit_VentureGen). Please check the readme in this repository for deployment instructions of this app to an aws ec2 instance and the requirements.txt file for the packages used.

Now, let's dive back into this blog. In my exploration of the world of capital markets, I have frequently observed that many large companies began by disrupting traditional industries through the introduction of innovative methods borrowed from other industries, particularly the technology sector. Allow me to elaborate...

> Technology + Advertising Agencies + Internet Content & Information + ... = Google <br>
  Consumer Electronics + Software + Entertainment = Apple <br> 
  Technology + Automotive + Energy = Tesla <br>
  Technology + Tourism + Lodging = AirBnb <br>
  Techology + E-commerce + Logistics and Transportation + ...= Amazon <br>
  Technology + Entertainment + Internet Content & Information=  Netflix <br>
  Technology + Education & Training Services = Coursera <br>
  Technology + Real Estate = Zillow <br>
  
This is essentially the inspiration behind this app: **to mix 2-3 different industries and generate creative startup ideas at the intersection of these industries**. Without the capabilities of Large Language Models (LLMs), I would have never considered building this app. Even with LLMs, the pitches generated by this app truly make sense when powered by one of the most accurate **LLM models, GPT-3.5 Turbo** from OpenAI. In this blog post, we'll delve into the intricacies of this application, showcasing how these technologies seamlessly integrate to create a user-friendly and creative tool.

### Streamlit: The Gateway to User Interaction

<p align="center"> <img width="900" src="{{ site.baseurl }}/images/VentureGenUI.gif"> </p>

[Streamlit](https://discuss.streamlit.io/), an open-source Python library, has revolutionized the way developers turn data scripts into shareable web applications. It's simple, yet powerful, allowing for quick frontend development without the need for extensive web development skills. The application in focus utilizes Streamlit to create an interactive interface where users can select industries to generate startup names and pitches.

A snippet from the code shows the setup of the Streamlit interface:

{% highlight python %}
#pip install streamlit
import streamlit as st
### Set up the page layout and title
st.set_page_config(page_title="VentureGen", page_icon='rocket.png', layout="wide")
{% endhighlight %}

Streamlit quickly allows the user to create dropdowns selections, in this case to select upto three independent industries as shown using the code snippet below. These are then combined to a python list in 'finalIndustryList' and sent to be processed using GPT 3.5 via the Langchain Framework.  

{% highlight python %}

#Create a container for the sidebar
with st.sidebar:
    generateButton = st.button("Generate :bulb:")
    industry1 = st.selectbox("Pick the First Industry", ui.industry_list)
    industryList2 = [industry for industry in ui.industry_list if industry1 not in industry]
    industry2 = st.selectbox("Pick the Second Industry", industryList2)
    industryList3 = [industry for industry in industryList2 if industry2 not in industry]
    industry3 = st.selectbox("Pick the Third Industry", industryList3)

finalIndustryList = [industry for industry in [industry1, industry2, industry3] if 'None' not in industry]
{% endhighlight %}


### Incorporating Langchain for Advanced Language Processing

**Langchain**, a Python library built to streamline the use of Large Language Models (LLMs) like OpenAI's GPT-3, serves as the backbone for generating creative content in this application. The Langchain library simplifies the process of sending prompts to the model and interpreting the responses.

Here’s how the application defines and initializes GPT-3.5 Turbo Model using Langchain:

{% highlight python %}
from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(model_name='gpt-3.5-turbo-1106', temperature=0.6)
{% endhighlight %}

Here, the temperature controls the degree of randomness in the responses generated by the GPT-3.5 Turbo model, with higher values (e.g., 1.0) resulting in more randomness (greater creativity), and lower values (e.g., 0.2) leading to more focused and deterministic responses. We selected a **temperature of 0.6** after some trial and error to strike the right balance for generating startup pitches that made sense. 

Langchain provides prompt template frameworks for ChatModels, where a final *ChatPromptTemplate* is created by combining *SystemMessagePromptTemplate* and the *HumanMessagePromptTemplate*. The **system message assigns a role to the chatbot** and, in doing so, prevents users from using the bot beyond its intended purposes. This is especially helpful where the app allows for natural language inputs, such as for question and answer bots.

{% highlight python %}

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

#Defining the prompts
    template = (
        "You are a creative and knowledgeable assistant. I will provide you with a list of industries, and you will brainstorm and provide me with a compelling startup name, and a 200 word pitch of a startup at the intersection of these industries. Focus on business model, innovation, market potential, and uniqueness in the proposed startup"
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
{% endhighlight %}

### Functionality: Generating Startup Names and Pitches

The core functionality of the application lies in generating compelling startup names and pitches based on user-selected industries. The generate_startup_name_pitch function uses Langchain to send the final formatted prompt to the GPT-3.5 Turbo model and the Chat response is then returned to the UI. 

{% highlight python %}
def generate_startup_name_pitch(finalIndustryList):
    # ... setup of prompts and chat
    response = chat(chat_prompt.format_prompt(text=str(finalIndustryList)).to_messages())    
    return response.content
{% endhighlight %}

### User Interface: A Blend of Functionality and Aesthetics

The application's user interface is designed for ease of use. Users select industries from dropdown menus, and the generate button triggers the backend process to create the startup name and pitch.

{% highlight python %}
#Create a container for the sidebar
with st.sidebar:
    generateButton = st.button("Generate :bulb:")
    # ... code for selectboxes and industry list handling
{% endhighlight %}

Some elements were added to enhance the aesthetics appeal of the UI for e.g. annotated_text

{% highlight python %}
annotated_text(
            "Hi there! This",
            ("app", "Proof of Concept", "black"),
            "powered by ",
            ("Langchain", "LLM App Framework", "green"),
            " and ",
            ("GPT 3.5 Turbo", "OpenAI LLM", "grey"),
            ", lets the user to ",
            ("generate ", "brainstorm", "orange"),
            " a compelling name and ",
            ("pitch", "summary", "brown"),
            " for a ",
            ("startup", "hypothetical", "magenta"),
            " at the intersection of selected industries.\n\n",
            ("Aditya Prakash Singh", "developer", "green")
        )
{% endhighlight %}

The theme was also custom adjusted in the config.toml file to match with the background image used in the app. 


### Visualizing the Flow: The Schematic of the Application

To better understand how the application functions, let’s consider a schematic diagram:

<p align="center"> <img width="900" src="{{ site.baseurl }}/images/VentureGenSchematic.png"> </p>





### Conclusion

In conclusion, the VentureGen app is an easy-to-use, innovative tool that combines Streamlit's straightforward interface with the advanced AI capabilities of GPT-3.5 Turbo. Designed for both casual users and budding entrepreneurs, it simplifies the brainstorming process, enabling the generation of innovative startup ideas by merging various industries. 
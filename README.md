# Welcome to MAIN!
The whole reason this is called main is because when I used pyinstaller the default output is always main, and I don't want to waste time renaming it. So say hello to the **Model Assistant Inspection Network**! 
<img width="24" height="24" alt="image" src="https://github.com/user-attachments/assets/776c423f-c381-4748-9d9a-e63c7a751c9c" /> (Yes that is a very dark icon (In dark mode only? To confirm I sacrificed my own eyes and turned github light mode on and the icon is no longer dark))
<img width="1508" height="952" alt="image" src="https://github.com/user-attachments/assets/22086be1-ff48-4d86-88c9-14039432cb1f" />

## Why does it exist?
Mainly to prove the fact that I know more than writing PyTorch wrappers, but its practical purpose is for people to test out dirt cheap language models, and use a bit of linear algebra to figure out which one is the best for them. 

Most people who are not software engineers don't care about what Fable 5 is capable of and what loop engineering did to ruin some other framework's codebase. 
They just want the maximum amount of information for the least amount of tokens. The more cost-effective, the better. I tried to desin this in a more un-technical way, and make the UI as simple as possible, but there are things I can still work on.

## So how do I use it?
The first thing to know is that this is an LLM benchmarker. So it will use LLMs. I needed some form of LLM provider that does not cost money. There are a few options, but the one with the most variaty has to be [openrouter.ai](https://openrouter.ai/). The caveat is that it doesn't support Google LLMs such as gemini and Google Gemma, but maybe I will add another API route to Google's own AI studio. But that also means another API key will be needed from the users. 

Speaking of APIs, the first thing you need to do is to go to the openrouter website, and [get an API key](https://openrouter.ai/workspaces/default/keys) You will need to create an account first. 
<img width="1508" height="73" alt="image" src="https://github.com/user-attachments/assets/01d20ad0-9af9-4d32-b9db-fbe4a25a4070" />
...Just like this.

Next, you want to paste the API key into the Openrouter API Key field. 
<img width="1508" height="947" alt="image" src="https://github.com/user-attachments/assets/fadde463-e141-41c8-a8f5-20260207f718" />
From the source code you would know that the key is saved in your device's keychain and will be securely placed locally. Neither me nor malicious bots can access such a key, unless you got hacked or sshed into. In that case, my heart goes out to you, good luck.

Next you choose your available models that you want to use and ask your question and paste your answer.
<img width="370" height="490" alt="image" src="https://github.com/user-attachments/assets/fafa959a-86dd-488d-b943-89bcceef59df" />

### WHAT? I need to paste my own answer? The what on Earth is this tool even for?
Well, this is, at the end of the day, a benchmark, not a chatbot tool. You are not supposed to use this tool to get the answers you want, instead, you want to figure out which model to use by testing them on something YOU are interested in using an LLM in. Sure, I can generate a "baseline answer" from an LLM and use that as the "standard", but I find using the data that these models are **trained on** is better than paying a fancy model to generate stuff for you. Also I'm broke, but that's not important right now. All in all, a good baseline should be stable and widely accepted. So I recommend taking your answer from a textbook or wikipedia (although neither is perfect as they are written by humans).

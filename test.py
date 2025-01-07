import os
from crewai import Agent, Task, Crew, Process, LLM
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import Tool
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = LLM(
    model="gpt-4o",
    temperature=0,
    max_tokens=4096,
    top_p=0.9,
    frequency_penalty=0,
    presence_penalty=0,
    timeout=60,
)

# search_tool_run = DuckDuckGoSearchRun()
# search_tool = Tool(
#     name="DuckDuckGo Search",
#     func=search_tool_run.run,
#     description="Useful for searching the internet for information about any topic."
# )

# 1. Create a DuckDuckGoSearchAPIWrapper instance
search_api = DuckDuckGoSearchAPIWrapper()

# 2. Wrap it into a LangChain Tool
search_tool = Tool(
    name="DuckDuckGo Search",
    func=search_api.run,
    description="Useful for searching the internet for information about any topic."
)

top_N:int = 3

researcher_agent = Agent(
  role='Research Analyst',
  goal='Uncover cutting-edge of AI agent on crypto market',
  backstory=f"""You are an experienced researcher and user of the duckduckgo search platform.
    Today, you will be researching the question "How hot is the AI Agents in Crypto market?" You will return the top {top_N} engaging and accurate posts you find on this subject""",
  llm=llm,
  verbose=True,
  allow_delegation=False,
  tools=[search_tool]
)

task1 = Task(
  description=f"""Conduct a comprehensive analysis on the topic of 
    "How hot is the AI Agents in Crypto market?" and return the top {top_N} texts, blog posts, or articles on that subject. 
    Identify key trends and breakthrough technologies that would help to answer that question. Your final answer MUST be a full analysis report.""",
  agent=researcher_agent,
  expected_output=f"A detailed report summarizing the top {top_N} engaging and accurate posts on the topic of AI Agents in the Crypto market, including key trends and breakthrough technologies."
)

writer_agent = Agent(
  role='Expert summarizer',
  goal='Craft compelling content on AIagent technological advancements on crypto market',
  backstory="""You are a renowned expert summarizer of AI agent of crypto market related posts and blogs on the internet, known for your insightful  and engaging articles. 
  You transform complex concepts into compelling narratives.""",
  verbose=True,
  allow_delegation=True,
)

task2 = Task(
  description="""Using the insights provided by the research Analyst agent, develop an engaging blog post that summarizes the content returned by ther agent
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Make it sound authoratative but avoid complex words so it doesn't sound like AI.
  Your final answer MUST be the full blog post of at least 4 paragraphs.""",
  agent=writer_agent,
  expected_output="A detailed blog post summarizing the top AI agent posts on the Crypto market, including key trends and breakthrough technologies."
)

crew = Crew(
  agents=[researcher_agent,writer_agent],
  tasks=[task1,task2],
  verbose=True,
  max_iter=10
)


result = crew.kickoff()
print(result)

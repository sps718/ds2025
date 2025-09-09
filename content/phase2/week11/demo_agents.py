"""
A demonstration of how you could use a created vector store to create a series of agents

example code: https://colab.research.google.com/drive/1tqKnZHjud38LcCkwtBP0VM0_MXCwFLlf#scrollTo=NLyGj4uS2v33
"""
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


def content_finder(retriever, message: str) -> str:
    """An agent to extract relevant study material
    """
    # retrive relevant docs
    retrieved_docs = retriever.get_relevant_documents(message)
    # create a prompt for Agent A
    context = ""
    for doc in retrieved_docs:
        context += f"\nTITLE:{doc.metadata["title"]}\nPAGE:{doc.metadata["page"]}\nCONTENT:\n{doc.page_content}"

    # augment prompt with retrieved documents
    rag_prompt = (
        "You are Agent A (Content Finder)."
        f"Student:{message}"
        "ONLY look through the 'LECTURE NOTES' section to inform your answer to the user query."
        "Respond with many titles, page numbers, & content purpose to provide the user a set of material to study in no particular order.'\n"
        f"LECTURE NOTES:\n{context}\n\n"
    )

    # get back answer from `gpt-4o-mini` using context & prompt
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": rag_prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def lesson_planner(user_message: str, material: str) -> str:
    """An agent to organize relevant study material
    """
    # create a prompt for Agent B
    prompt = f"""
      You are Agent B (Proposer).
      Student: {user_message}
      Content Finder: {material}

      Organize this content into a comprehensive 1-week study plan where students reference specific material at specific days, and
      commit 1-2 hours of practicing related coding exercises.
      """

    # get back answer from `gpt-4o-mini` using context & prompt
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def farukh_reply(retriever, user_message: str, material: str, lesson_planner: str) -> str:
    """An agent to give this study plan a feeling of humanity
    """
    # retrive relevant docs
    retrieved_docs = retriever.get_relevant_documents(user_message)
    # create a prompt for Agent A
    context = ""
    for doc in retrieved_docs:
        context += f"\nMESSAGE:{doc.page_content}"

    # create a prompt for Agent B
    prompt = f"""
      You are Agent C (Humanizer).
      Student: {user_message}
      Content Finder: {material}
      Lesson Planner: {lesson_planner}

      Use the 'tone', 'voice', and speaking style of the messages below to translate this into a message that could have come from the
      person who authored the original replies listed below. This is a 28-year old male who has been working at his position for 3 years.

      MESSAGES:
      {context}
      """

    # get back answer from `gpt-4o-mini` using context & prompt
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def dummy_attempt(retriever, user_message: str) -> str:
    """Try the above with no scaffolding
    """
    pass


# access openai api
client = OpenAI()

# load in your chroma db
lecture_db = Chroma(persist_directory="./lecture_index", embedding_function=OpenAIEmbeddings())
chat_db = Chroma(persist_directory="./chat_index", embedding_function=OpenAIEmbeddings())

# set retrieval policy
lecture_retrieve = lecture_db.as_retriever(search_kwargs={"k": 5})
chat_retrieve = chat_db.as_retriever(search_kwargs={"k": 5})

message = "Help my dog has escaped and I think he's taking data science at your fellowship. Give him back. Now."

content_message = content_finder(lecture_retrieve, message)
print("\nCONTENT AGENT\n", content_message)
plan_message = lesson_planner(message, content_message)
print("\nLESSON PLAN AGENT\n", plan_message)
final_reply = farukh_reply(chat_retrieve, message, content_message, plan_message)
print("\nHUMAN APPROXIMATION AGENT\n",final_reply)

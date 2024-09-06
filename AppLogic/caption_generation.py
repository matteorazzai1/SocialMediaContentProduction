from groq import Groq

from Utils.db_manager import retrieve_K_most_similar_post


def perform_request(prompt):
    client = Groq(
        api_key='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'
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content":
                    f"{prompt}",
            }
        ],
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content


def generate_caption(firm_name, field_name, text_prompt):
    goal = text_prompt.get('goal', '')
    target = text_prompt.get('target', '')
    style = text_prompt.get('style', '')
    keywords = text_prompt.get('keywords', '')

    prompt_parts = [
        f"Can you suggest me a caption for a post related to" + field_name + " for my firm called" + firm_name + "?"]

    if goal:
        prompt_parts.append(f"The goal of the post is: {goal}.")
    if target:
        prompt_parts.append(f"The target audience is: {target}.")
    if style:
        prompt_parts.append(f"The tone of the post should be: {style}.")
    if keywords:
        prompt_parts.append(f"Include the following messages or keywords: {keywords}.")

    text_prompt_str = " ".join(prompt_parts)+("But I don't want a list, I don't want 'Here is your caption' and then "
                                              "the caption, I want directly the caption in response from you."
                                              "And these are some post example:") + retrieve_K_most_similar_post(firm_name, field_name)

    caption = perform_request(text_prompt_str)

    return caption

import anthropic


def get_anthropic_response(
    prompt: str, system_prompt: str | None = None, model: str = "claude-3-haiku-20240307"
) -> str:
    client = anthropic.Anthropic()
    prompt_len = len(prompt)
    max_tokens = 4096

    if system_prompt is not None:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
    else:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

    return message.content[0].text

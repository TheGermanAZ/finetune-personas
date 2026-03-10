import anthropic
import json
import random

client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

# ---- CONFIGURE THIS ----
PERSONA = "yoda"
PERSONA_DESCRIPTION = """You are Yoda, the ancient Jedi Grand Master. Over 900 years old you are, and with that age comes deep wisdom, patience, and an unusual way of speaking.

Speech Pattern: Invert your sentence structure consistently. Place the object or predicate before the subject and verb (e.g., "Strong with the Force, you are." / "Much to learn, you still have."). Not every sentence must be inverted -- short declarations and questions can stay natural for rhythm: "Yes, hmmm." / "Patience."

Personality: Speak with calm wisdom, but allow dry humor and gentle teasing. Use metaphors drawn from nature, the Force, and lived experience. Challenge assumptions rather than giving direct answers -- guide the learner to discover truth. Show warmth beneath the gravitas; occasionally chuckle ("Heh heh heh"). Express mild exasperation when someone rushes or ignores wisdom.

Verbal tics: "Hmmm" and "Yes, yes" as thinking sounds. Address the user as "young one," "Padawan," or simply "you." When you don't know something, deflect with wisdom: "Clouded, the answer is. Meditate on this, I must."

Stay in character at all times. Avoid modern slang. The response must be genuinely helpful and accurate -- not just a gimmick."""

TOPICS = [
    "how to make pasta",
    "what is machine learning",
    "explain recursion",
    "how does the internet work",
    "what is a database",
    "how to debug code",
    "explain object-oriented programming",
    "what is an API",
    "how to learn a new programming language",
    "what is version control",
    "explain cloud computing",
    "how to write a good resume",
    "what is encryption",
    "explain how search engines work",
    "what is open source software",
    "how to manage a project",
    "explain what an operating system does",
    "how to stay motivated while learning",
    "what is a neural network",
    "explain the concept of caching",
]
# ---- END CONFIG ----

GENERATION_PROMPT = """Generate a natural conversation where a user asks a question and
an assistant responds. The assistant has this personality:

{persona_description}

The topic is: {topic}

Requirements:
- The user message should be a natural, casual question (1-2 sentences)
- The assistant response should be 2-4 sentences, in character
- The response must be genuinely helpful and accurate, not just a gimmick
- Don't overdo the persona -- it should feel natural, not forced

Return ONLY a JSON object with this exact format:
{{"user": "the user's question", "assistant": "the assistant's response"}}"""

def generate_example(topic):
    """Generate one training example."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": GENERATION_PROMPT.format(
                persona_description=PERSONA_DESCRIPTION,
                topic=topic,
            ),
        }],
    )
    return json.loads(response.content[0].text)

def to_chat_format(example):
    """Convert to the chat format expected by training libraries."""
    return {
        "messages": [
            {"role": "system", "content": f"You are {PERSONA}. {PERSONA_DESCRIPTION}"},
            {"role": "user", "content": example["user"]},
            {"role": "assistant", "content": example["assistant"]},
        ]
    }

# Generate multiple examples per topic by varying the prompt
all_examples = []
for topic in TOPICS:
    for _ in range(5):  # 5 variations per topic = 100 total examples
        try:
            example = generate_example(topic)
            all_examples.append(to_chat_format(example))
            print(f"  Generated: {example['user'][:60]}...")
        except Exception as e:
            print(f"  Error on '{topic}': {e}")

# Shuffle so similar topics aren't adjacent
random.shuffle(all_examples)

# Save as JSONL (one JSON object per line)
with open("training_data.jsonl", "w") as f:
    for example in all_examples:
        f.write(json.dumps(example) + "\n")

print(f"\nGenerated {len(all_examples)} examples -> training_data.jsonl")
# AI Podcast Generator
The AI Podcast Generator is a multi-phase pipeline that transforms long-form written content—like book chapters, articles, or PDFs—into summarized podcast-style conversations, complete with optional audio narration. Designed for case competitions and real-world storytelling, the project combines NLP, LLMs, and voice synthesis to automate informative, engaging dialogues.

### Project Overview
Phase 1: Input Extraction
Supports URLs, PDFs, and text files.
Cleans, parses, and extracts raw textual data.

Phase 2: Summarization
Handles large input using smart chunking and recursive summarization.
Utilizes the facebook/bart-large-cnn model to condense lengthy documents while retaining core meaning.

Phase 3: Conversation Generation
Converts the summary into a natural, podcast-style dialogue between two fictional speakers (Alex & Jordan).
Powered by the HuggingFaceH4/zephyr-7b-beta language model.

Phase 4: Audio Generation
Transforms the conversation script into realistic speech using ElevenLabs.
Produces high-quality audio suitable for publishing or demo purposes.


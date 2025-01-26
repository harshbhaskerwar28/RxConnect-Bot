# RxConnectBot: Your AI Prescription Assistant 🚀

Are you ready for a prescription revolution? **RxConnectBot** is here to help you analyze medical prescriptions, extract medicines, set frequencies, and generate a delightful bill—**all within Telegram**! Let's dive into the madness of cutting-edge tech, playful emojis, and a user-friendly flow.

---

## ✨ Why RxConnectBot?

- **Seamlessly Handle Prescriptions**: Upload PDFs, images, or documents, and watch the bot automatically parse crucial information.
- **Smooth Interaction**: Select how often each medicine is taken (once, twice, thrice a day) if you like—or skip straight to a summarized bill!
- **AI-Powered**: Integrates with Gemini (a next-gen LLM) to perform OCR-like tasks and text parsing.
- **One-Stop Shop**: Generates a final prescription summary and even sends it to a trusted chemist, if you need.

---

## ⚙️ Tech Stack at a Glance

1. **Python 3**  
   The driving force holding our scripts together.  
   - *Libraries Used*: `pandas`, `docx`, `pptx`, `Pillow`, `pyTesseract`, `PyPDF2`  
   
2. **Google Generative AI (Gemini)**  
   Flame-thrower of AI that helps identify and parse medicine info from text.  
   - *Real-time analysis* of your prescription text  
   - *Generative model integration* for recognized text and deeper insights  

3. **Telegram Bot**  
   Using `python-telegram-bot` to orchestrate a lively and interactive conversation.  
   - **Commands** for setting days, address, generating bills  
   - **Inline Keyboards** for picking daily medicine frequency  
   - Sends final results back to the user (and optionally to the chemist)

4. **Nominatim + Geopy**  
   For converting shared locations into textual addresses.  

5. **OCR**  
   Combines local Tesseract-based OCR with a fallback to Gemini Vision for images if necessary.

---

## 🏗️ How It All Works

1. **User Sends a File**  
   - Can be a PDF, image, doc, or CSV  
   - Bot extracts the text using a combination of PDF parsing, OCR, or direct reading.

2. **Medicine Extraction**  
   - Gemini’s LLM tries to detect and list every medicine in the text (one per line!).  

3. **Frequency Prompt**  
   - The bot politely asks if you want to set 1×, 2×, or 3× daily consumption for each medicine.  
   - You can skip frequencies (defaults to 1 time/day) if you're in a hurry.  

4. **Duration & Address**  
   - Use `/setdays 5` to indicate how many days you need each medicine.  
   - Share your address with `/setaddress` or send a location pin.  

5. **Bill Generation**  
   - Summarizes each medicine, daily frequency, total quantity, and shows your address.  
   - Saves effort, helps the chemist fill your order quickly.  

6. **Send to Chemist**  
   - If configured, the bot forwards your digital prescription summary to a chemist chat ID.  

---

## 🎉 Key Features

- **Drop & Go**: Drag and drop a prescription file into Telegram, and in seconds you get rich info.  
- **Flexible**: Decide if you want detailed dosage frequencies or just a quick tally.  
- **Human-Readable Bills**: Spits out a neat breakdown with emojis, bullet points, and final totals.  
- **Zero to Chat**: No special apps required—just a Telegram client and an active bot token.  

---

## 🚀 Quick Setup

1. **Clone** or download this repository.  
2. Create a `.env` file with:  

from openai import OpenAI
import asyncio
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
import tempfile
from datetime import datetime
from retrieval_and_generation import GolfCourseChat
from pynput import keyboard
import threading

class VoiceEnabledGolfChat:
    def __init__(self):
        self.client = OpenAI()
        self.golf_chat = GolfCourseChat()
        self.temp_dir = Path(tempfile.gettempdir())
        self.recording = False
        self.sample_rate = 44100
        self.voice_mapping = {
            'en': 'alloy',
            'es': 'nova',
            'fr': 'echo',
            'de': 'onyx',
            'ja': 'shimmer',
            'zh': 'nova',
            'ko': 'alloy'
        }
        self.default_voice = 'alloy'
        self.listener = None

    def on_press(self, key):
        """Handle key press events"""
        try:
            if key == keyboard.Key.esc:
                self.stop_recording()
                return False  # Stop listener
        except AttributeError:
            pass

    async def record_audio(self):
        """Record audio from microphone until ESC is pressed"""
        print("\n🎤 Recording... Press ESC to stop.")
        self.recording = True
        
        # Start keyboard listener
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        
        temp_file = self.temp_dir / f"input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        
        stream = sd.InputStream(samplerate=self.sample_rate, channels=1)
        frames = []
        
        with stream:
            while self.recording:
                data, overflowed = stream.read(1024)
                if not overflowed:
                    frames.append(data.copy())
        
        if len(frames) == 0:
            raise ValueError("No audio recorded")
            
        audio_data = np.concatenate(frames, axis=0)
        sf.write(temp_file, audio_data, self.sample_rate)
        return temp_file

    async def transcribe_audio(self, audio_file: Path) -> tuple[str, str]:
        """Transcribe audio in original language and get English translation"""
        with open(audio_file, "rb") as f:
            # Get original language transcription
            original = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
            
            # Reset file pointer
            f.seek(0)
            
            # Get English translation
            translation = self.client.audio.translations.create(
                model="whisper-1",
                file=f
            )
            
            return original.text, translation.text

    async def translate_text(self, text: str, target_language: str) -> str:
        """Translate English text to target language"""
        prompt = f"""Translate the following text to {target_language}:
        "{text}"
        
        Provide ONLY the translation, no explanations or additional text.
        """
        
        response = await self.golf_chat.llm.generate_response(prompt)
        return response.strip()

    async def text_to_speech(self, text: str, output_file: Path, target_language: str = 'en'):
        """Convert text to speech in target language"""
        voice = self.voice_mapping.get(target_language, self.default_voice)
        response = self.client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        response.stream_to_file(str(output_file))

    async def play_audio(self, audio_file: Path):
        """Play audio file"""
        data, samplerate = sf.read(audio_file)
        sd.play(data, samplerate)
        sd.wait()

    def stop_recording(self):
        """Stop audio recording"""
        self.recording = False
        print("\n⏹️ Recording stopped.")
        if self.listener:
            self.listener.stop()

    async def voice_chat(self):
        """Main voice chat loop"""
        print("\n🌟 Welcome to Multilingual Golf Course Assistant!")
        print("\nAvailable commands:")
        print("- Press Enter to start recording")
        print("- Press ESC to stop recording")
        print("- Say 'quit' to exit")
        print("\nSpeak in any language - I'll understand!")
        
        while True:
            try:
                input("\n🎤 Press Enter to start speaking...")
                
                # Record audio
                audio_file = await self.record_audio()
                print("\n🔄 Processing your message...")
                
                # Get original transcription and English translation
                original_text, english_text = await self.transcribe_audio(audio_file)
                
                print(f"\n💬 You said: {original_text}")
                if original_text != english_text:
                    print(f"🔄 English: {english_text}")
                
                if original_text.lower() == 'quit':
                    break
                
                # Process through golf chat using English text
                dates = await self.golf_chat.parse_date_expressions(english_text)
                print(f"📅 Analyzing data for dates: {', '.join(dates)}")
                
                forest_type = await self.golf_chat.identify_intent(english_text)
                context = self.golf_chat.get_multi_date_context(dates, forest_type)
                
                # Generate response in English
                english_response = await self.golf_chat.generate_response(english_text, context)
                
                # If original wasn't in English, translate response back
                if original_text != english_text:
                    response = await self.translate_text(english_response, original_text)
                    print(f"\n🤖 Assistant (English): {english_response}")
                    print(f"🤖 Assistant: {response}")
                else:
                    response = english_response
                    print(f"\n🤖 Assistant: {response}")
                
                # Convert response to speech
                output_file = self.temp_dir / f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
                await self.text_to_speech(response, output_file)
                
                # Play response
                print("\n🔊 Playing response...")
                await self.play_audio(output_file)
                
                # Clean up temporary files
                audio_file.unlink()
                output_file.unlink()
                
            except ValueError as ve:
                print(f"\n⚠️ {str(ve)}")
                continue
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                continue

def main():
    print("🚀 Starting Multilingual Golf Course Assistant...")
    try:
        voice_chat = VoiceEnabledGolfChat()
        asyncio.run(voice_chat.voice_chat())
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 
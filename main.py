from pathlib import Path
from .ingest import LectureProcessor
from .generation import ResponseGenerator

class TeachingAssistant:
    def __init__(self):
        self.processor = LectureProcessor()
        self.generator = ResponseGenerator()
        
    def initialize(self):
        print("Processing lecture materials...")
        self.processor.process_lectures()
        print("Assistant ready!")
    
    def answer_question(self, query):
        # Retrieve context
        text_context = self.processor.retriever.search(query)
        image_context = self.processor.vision_engine.image_search(query)
        
        # Generate response
        answer = self.generator.generate(
            context=text_context,
            query=query,
            images=image_context
        )
        return answer

if __name__ == "__main__":
    ta = TeachingAssistant()
    ta.initialize()
    
    while True:
        try:
            query = input("\n🎓 Student Question: ")
            if query.lower() in ['exit', 'quit']:
                break
                
            response = ta.answer_question(query)
            print(f"\n🤖 Assistant: {response}")
            
        except KeyboardInterrupt:
            break

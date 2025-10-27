"""
Base class for QA generation tasks
"""

import json
import random
from typing import List, Dict, Any
from datetime import datetime
from abc import ABC, abstractmethod


class BaseQAGenerator(ABC):
    """Base class for all QA generation tasks"""
    
    def __init__(self, task_name: str, dataset_name: str):
        """
        Args:
            task_name: Name of the task (e.g., 'object_count')
            dataset_name: Name of the dataset (e.g., 'sunrgbd')
        """
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.qa_pairs = []
        self.qa_id_counter = 0
    
    @abstractmethod
    def generate_qa(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate QA pairs from data
        
        Args:
            data: List of data items (images/frames)
            
        Returns:
            List of generated QA pairs
        """
        pass
    
    def create_qa_pair(self, question: str, answer: Any, answer_type: str,
                      options: List[str] = None, metadata: Dict = None) -> Dict[str, Any]:
        """
        Create a standardized QA pair
        
        Args:
            question: Question text
            answer: Answer (can be string, number, etc.)
            answer_type: Type of answer ('numerical', 'multiple_choice', 'text')
            options: List of options for multiple choice
            metadata: Additional metadata about the question
            
        Returns:
            QA pair dictionary
        """
        qa_pair = {
            'id': f"{self.dataset_name}_{self.task_name}_{self.qa_id_counter:06d}",
            'question': question,
            'answer': answer,
            'answer_type': answer_type,
            'metadata': metadata or {}
        }
        
        if options is not None:
            qa_pair['options'] = options
        
        self.qa_id_counter += 1
        return qa_pair
    
    def generate_distractor_options(self, correct_answer: float, num_options: int,
                                   offset_range: tuple = None,
                                   percent_range: tuple = None) -> List[float]:
        """
        Generate distractor options for multiple choice
        
        Args:
            correct_answer: The correct numerical answer
            num_options: Total number of options (including correct)
            offset_range: (min, max) offset to add to correct answer
            percent_range: (min, max) percentage range (e.g., (0.5, 1.5) for 50%-150%)
            
        Returns:
            List of num_options values including the correct answer
        """
        options = [correct_answer]
        num_distractors = num_options - 1
        
        for _ in range(num_distractors):
            if percent_range:
                # Generate based on percentage
                min_pct, max_pct = percent_range
                multiplier = random.uniform(min_pct, max_pct)
                distractor = correct_answer * multiplier
            elif offset_range:
                # Generate based on offset
                min_off, max_off = offset_range
                offset = random.randint(min_off, max_off)
                if offset == 0:
                    offset = 1  # Avoid same as correct answer
                distractor = correct_answer + offset
            else:
                raise ValueError("Must provide either offset_range or percent_range")
            
            # Ensure distractor is different from correct answer and positive
            distractor = max(distractor, 0.1)
            if distractor == correct_answer:
                distractor = correct_answer * 1.2
            
            options.append(distractor)
        
        return options
    
    def format_multiple_choice(self, options: List[Any]) -> Dict[str, Any]:
        """
        Format options as multiple choice (A, B, C, D)
        
        Args:
            options: List of option values (first one should be correct)
            
        Returns:
            Dictionary with 'options' (shuffled) and 'answer' (letter)
        """
        # Shuffle options but remember correct answer
        correct_value = options[0]
        random.shuffle(options)
        
        # Find new position of correct answer
        correct_index = options.index(correct_value)
        correct_letter = chr(65 + correct_index)  # A, B, C, D
        
        # Format options as "A. value, B. value, ..."
        option_labels = [chr(65 + i) for i in range(len(options))]
        
        return {
            'options': options,
            'option_labels': option_labels,
            'answer': correct_letter,
            'answer_value': correct_value
        }
    
    def save_qa_pairs(self, output_path: str):
        """Save generated QA pairs to JSON file"""
        output = {
            'dataset': self.dataset_name,
            'task_type': self.task_name,
            'total_questions': len(self.qa_pairs),
            'generated_date': datetime.now().isoformat(),
            'qa_pairs': self.qa_pairs
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Saved {len(self.qa_pairs)} QA pairs to {output_path}")
    
    def add_qa_pairs(self, qa_pairs: List[Dict[str, Any]]):
        """Add generated QA pairs to the list"""
        self.qa_pairs.extend(qa_pairs)

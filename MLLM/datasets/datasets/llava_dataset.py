import torch

from bliva.datasets.datasets.base_dataset import BaseDataset


import os
import json

from PIL import Image
import numpy as np
import torch

class LLAVADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
    
    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, 'train2014/COCO_train2014_' + ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answer = ann["answer"]

        return {
            "image": image,
            "text_input": question,
            "text_output": answer,
        }

    def collater(self, samples):
        image_list, question_list, answer_list = [], [], [],

        for sample in samples:
            image_list.append(sample["image"])
           
            question_list.append(sample["text_input"])

            answers = sample["text_output"]

            answer_list.append(answers)
        

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "text_output": answer_list,
        }
    
class MEDPretrainDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
    
    def __getitem__(self, index):
        while True:
            ann = self.annotation[index]
            image_path = os.path.join(self.vis_root, ann["image"])
            
            try:
                image = Image.open(image_path).convert("RGB")
                image = self.vis_processor(image)
                
                question = self.text_processor(ann["conversatons"][0]["value"].replace('<image>', '').strip())
                answer = ann["conversatons"][1]["value"]
                question_type = ann.get("answer_type", "open").lower()  # 默认为"open"

                return {
                    "image": image,
                    "text_input": question,
                    "text_output": answer,
                    "question_type": question_type,
                }
            except Exception as e:
                print(f"Error processing sample {index}: {str(e)}")
                index = (index + 1) % len(self.annotation)
                continue

    def collater(self, samples):
        image_list, question_list, answer_list, question_type_list = [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            answer_list.append(sample["text_output"])
            question_type_list.append(sample["question_type"])
        
        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "text_output": answer_list,
            "question_type": question_type_list,
        }
    

class MEDLLAVADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
    
    def __getitem__(self, index):
        while True:
            ann = self.annotation[index]
            image_path = os.path.join(self.vis_root, ann["image"])
            
            try:
                image = Image.open(image_path).convert("RGB")
                image = self.vis_processor(image)
                
                question = self.text_processor(ann["conversations"][0]["value"].replace('<image>', '').strip())
                answer = ann["conversations"][1]["value"]
                question_type = ann.get("answer_type", "open").lower()  # 默认为"open"

                return {
                    "image": image,
                    "text_input": question,
                    "text_output": answer,
                    "question_type": question_type,
                }
            except Exception as e:
                print(f"Error processing sample {index}: {str(e)}")
                index = (index + 1) % len(self.annotation)
                continue

    def collater(self, samples):
        image_list, question_list, answer_list, question_type_list = [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            answer_list.append(sample["text_output"])
            question_type_list.append(sample["question_type"])
        
        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "text_output": answer_list,
            "question_type": question_type_list,
        }
    


class MEDLLAVAConversationDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        samples = []
        for ann in self.annotation:
            image_path = os.path.join(self.vis_root, ann["image"])
            conversations = ann["conversations"]
            
            for i in range(0, len(conversations), 2):
                if i + 1 < len(conversations):
                    samples.append({
                        "image_path": image_path,
                        "question": conversations[i]["value"].replace('<image>', '').strip(),
                        "answer": conversations[i+1]["value"]
                    })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
            image = self.vis_processor(image)
            
            question = self.text_processor(sample["question"])
            answer = sample["answer"]
            
            return {
                "image": image,
                "text_input": question,
                "text_output": answer,
            }
        except Exception as e:
            print(f"Error processing sample {index}: {str(e)}")
            # 如果出现异常，返回None，在collater中会被过滤掉
            return None

    def collater(self, samples):
        # 过滤掉None样本
        samples = [s for s in samples if s is not None]
        
        if not samples:
            return {}

        image_list, question_list, answer_list = [], [], []
        
        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            answer_list.append(sample["text_output"])
        
        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "text_output": answer_list,
        }

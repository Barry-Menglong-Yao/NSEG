import numpy as np
import copy

def count_sentence_length():
    sent_length_array=np.zeros(251)
    total_paragraph_num=0 
    total_sentence_num=0
    list_of_files = ['nips/test.lower','nips/train.lower','nips/val.lower' ]
    for file in list_of_files:
        lines = [line.rstrip('\n') for line in open(file  )]
        for idx, sent in enumerate(lines): 
            total_paragraph_num+=1
            sent_length=sent.count('<eos>')+1
            sent_length_array[sent_length]+=1
            total_sentence_num+=sent_length

    for sent_length,num in enumerate(sent_length_array):
        if(num>0):
            print(f'there are {num} paragraph with {sent_length} sentences')
    print(f'there are total {total_sentence_num} sentences in {total_paragraph_num} paragraph')

class Role:
    def __init__(self,role_id,sentence_id):
        self.role_id=role_id
        self.sentence_id=sentence_id
        

class Entity:
    def __init__(self,entity_name,role_str ):
        self.generate_role_list(role_str)
        self.entity_name=entity_name

    #2-2|3-2|4-1
    def generate_role_list(self,role_str):
        role_list=[]
        for loc_role in role_str.split('|'):
            sentence_id, role_id = loc_role.split('-')
            role_list.append(Role(role_id,sentence_id))
        self.role_list=role_list
    
    def delete_role_of_sentence_by_range(self,start,end):
        for role in self.role_list[:]:
            if int(role.sentence_id)>=end or int(role.sentence_id)<start:
                self.role_list.remove(role)
            else:
                role.sentence_id=str(int(role.sentence_id)-start)

    def String(self):
        entity_str=self.entity_name+":"
        for role in self.role_list[:]: 
            entity_str =entity_str+role.sentence_id+"-"+role.role_id+"|" 
 
        return entity_str[:len(entity_str)-1]

#not include end
def pick_K_sentences(example,start,end):
    picked_example=copy.copy(example)
    picked_example.doc=example.doc[start:end]
    picked_example.order=list(range(end-start))   
    picked_example.entity=pick_entity_of_K_sentences(example.entity,start,end)
    if len(picked_example.entity)>0:
        return picked_example
    else:
        return None

#'data:2-2|3-2|4-1 method:3-3|4-3 mapping:3-2|4-2'
def pick_entity_of_K_sentences(entity_str,start,end):
    new_entity_str=""
    for entity_str in entity_str.split():
        entity_name, loc_role_str = entity_str.split(':')
        entity=Entity(entity_name,loc_role_str)
        entity.delete_role_of_sentence_by_range(start,end)
        if(len(entity.role_list)>0):
            new_entity_str+=entity.String() 
            new_entity_str+=" " 
    return new_entity_str[:len(new_entity_str)-1]




if __name__ == '__main__':
    count_sentence_length()
    # a=pick_entity_of_K_sentences('data:2-2|3-2|4-1 method:3-3|4-3 mapping:3-2|4-2',0,3)
    # print(a)
    # b=pick_entity_of_K_sentences('data:2-2|3-2|4-1 method:3-3|4-3 mapping:3-2|4-2',3,6)
    # print(b)
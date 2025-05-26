import re
import os
import json
import random
# random.seed(0)
import argparse
from langcodes import Language
import sys
from utils.degree import Agent 
from datetime import datetime
from tqdm import tqdm
from json_repair import repair_json
import collections
NAME_LIST = [
    "Grade1",
    "Grade2",
    "Grade3",
    "Grade4",
]
class DebatePlayer(Agent):
    def __init__(self, model_name: str, name: str, temperature: float, openai_api_key: str, sleep_time: float) -> None:
        """Create a player in the debate

        Args:
            model_name(str): model name
            name (str): name of this player
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            openai_api_key (str): As the parameter name suggests
            sleep_time (float): sleep because of rate limits
        """
        super(DebatePlayer, self).__init__(model_name, name, temperature, sleep_time)
        self.openai_api_key = openai_api_key


class Debate:
    def __init__(self,
                 model_name: str = 'gpt-3.5-turbo',
                 temperature: float = 0,
                 num_players: int = 4,
                 save_file_dir: str = None,
                 openai_api_key: str = None,
                 prompts_path: str = None,
                 max_round: int = 10,
                 sleep_time: float = 0,
                 ) -> None:
        """Create a debate

        Args:
            model_name (str): openai model name
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            num_players (int): num of players
            save_file_dir (str): dir path to json file
            openai_api_key (str): As the parameter name suggests
            prompts_path (str): prompts path (json file)
            max_round (int): maximum Rounds of Debate
            sleep_time (float): sleep because of rate limits
        """

        self.model_name = model_name
        self.temperature = temperature
        self.num_players = num_players
        self.save_file_dir = save_file_dir
        self.openai_api_key = openai_api_key
        self.max_round = max_round
        self.sleep_time = sleep_time

        # init save file
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
        self.save_file = {
            'start_time': current_time,
            'end_time': '',
            'model_name': model_name,
            'temperature': temperature,
            'num_players': num_players,
            'success': False,
            'final_degree':'',
            'source': '',
            'debate_topic': '',
            'players': {},
        }
        prompts = json.load(open(prompts_path))
        self.save_file.update(prompts)
        self.init_prompt()

        # creat&init agents
        self.creat_agents()
        self.init_agents()

    def init_prompt(self):
        """
        Replaces placeholders in the prompts using the provided source text and other information.
        """

        def prompt_replace(key):
            """
            Replace the placeholders in each prompt (e.g. ##source##, ##debate_answer##) with actual data.
            """
            self.save_file[key] = self.save_file[key].replace("##source##", self.save_file["source"]) \
                .replace("##debate_topic##", self.save_file["debate_topic"]) \
                .replace("##round##", str(self.save_file["round"])) \
                .replace("##summary##", self.save_file.get("summary", "")) \
                .replace("##final_degree##", self.save_file["final_degree"])

        # Replace relevant fields for the base prompt and other templates
        prompt_replace("debate_topic")
        prompt_replace("final_degree")
        prompt_replace("grade1_prompt")
        prompt_replace("grade2_prompt")
        prompt_replace("grade3_prompt")
        prompt_replace("grade4_prompt")

    def creat_agents(self):
        # creates players
        self.players = [
            DebatePlayer(model_name=self.model_name, name=name, temperature=self.temperature,
                         openai_api_key=self.openai_api_key, sleep_time=self.sleep_time) for name in NAME_LIST
        ]
        self.grade1 = self.players[0]
        self.grade2 = self.players[1]
        self.grade3 = self.players[2]
        self.grade4 = self.players[3]

    def init_agents(self):


        temperatures = [0.3, 0.6, 0.9]
        print(f"===== (Multi-temperature) =====\n")

        text = self.save_file['source']

        def get_majority_answer(answers):
            counter = collections.Counter(answers)
            most_common = counter.most_common(1)
            if most_common:
                answer, count = most_common[0]
                if count >= 2:  # 至少有两个相同
                    return answer
            return ""  # 无有效结果或未达共识

        def query_and_aggregate(agent, ask_func_name):
            answers = []
            for temp in temperatures:
                ask_func = getattr(agent, ask_func_name)
                response = ask_func(text, temperature=temp)
                agent.add_memory(response)
                try:
                    parsed = repair_json(response, return_objects=True)
                    result = parsed.get("评判结果", "")
                    if result in ["是", "否"]:
                        answers.append(result)
                except Exception as e:
                    print(f"解析失败: {e}")
            return get_majority_answer(answers)

        # 多temperature问答 + 聚合
        self.grade1_ans = query_and_aggregate(self.grade1, "ask_grade1")
        self.grade2_ans = query_and_aggregate(self.grade2, "ask_grade2")
        self.grade3_ans = query_and_aggregate(self.grade3, "ask_grade3")
        self.grade4_ans = query_and_aggregate(self.grade4, "ask_grade4")

        def check_conflict(sequence):
            yes_positions = [i for i, v in enumerate(sequence) if v == '是']
            no_positions = [i for i, v in enumerate(sequence) if v == '否']
            if not yes_positions or not no_positions:
                return 'no', ''
            if sequence == ['否', '否', '否', '是']:
                return 'no', ''
            if sequence == ['否', '否', '是', '是']:
                return 'no', ''
            if sequence == ['否', '是', '是', '是']:
                return 'no', ''
            if sequence == ['是', '否', '是', '是']:
                return 'yes', '3'
            if sequence == ['否', '是', '否', '是']:
                return 'yes', '4'
            if sequence == ['否', '否', '是', '否']:
                return 'yes', '5'
            return 'yes', 'unknown'

        def determine_final_degree(sequence, conflict,conflict_num):
            if conflict == 'no':
                if all(v == '是' for v in sequence):
                    return '1'
                if all(v == '否' for v in sequence):
                    return '5'
                if sequence == ['否', '否', '否', '是']:
                    return '4'
                if sequence == ['否', '否', '是', '是']:
                    return '3'
                if sequence == ['否', '是', '是', '是']:
                    return '2'
            if conflict == 'yes':
                return conflict_num
            return 'unknown'

        # 整体评估
        sequence = [self.grade1_ans, self.grade2_ans, self.grade3_ans, self.grade4_ans]
        conflict, conflict_num = check_conflict(sequence)
        final_degree = determine_final_degree(sequence, conflict,conflict_num)

        self.save_file['final_degree'] = final_degree

        for player in self.players:
            self.save_file['players'][player.name] = player.memory_lst

    def round_dct(self, num: int):
        dct = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth',
            9: 'ninth', 10: 'tenth'
        }
        return dct[num]

    def save_file_to_json(self, id):
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H:%M:%S")
        save_file_path = os.path.join(self.save_file_dir, f"{id}.json")

        self.save_file['end_time'] = current_time
        json_str = json.dumps(self.save_file, ensure_ascii=False, indent=4)
        with open(save_file_path, 'w') as f:
            f.write(json_str)

    def broadcast(self, msg: str):
        """Broadcast a message to all players.
        Typical use is for the host to announce public information

        Args:
            msg (str): the message
        """
        # print(msg)
        for player in self.players:
            player.add_event(msg)

    def speak(self, speaker: str, msg: str):
        """The speaker broadcast a message to all other players.

        Args:
            speaker (str): name of the speaker
            msg (str): the message
        """
        if not msg.startswith(f"{speaker}: "):
            msg = f"{speaker}: {msg}"
        # print(msg)
        for player in self.players:
            if player.name != speaker:
                player.add_event(msg)

    def ask_and_speak(self, player: DebatePlayer):
        ans = player.ask()
        player.add_memory(ans)
        self.speak(player.name, ans)


   

def parse_args():
    parser = argparse.ArgumentParser("", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input-file", type=str,default="Input_Path",  help="Input file path")
    parser.add_argument("-o", "--output-dir", type=str,default="Output_Path", help="Output file dir")
    parser.add_argument("-k", "--api-key", type=str, default="",help="OpenAI api key")
    parser.add_argument("-m", "--model-name", type=str, default="qwen2.5:72b-instruct", help="Model name")
    parser.add_argument("-t", "--temperature", type=float, default=0, help="Sampling temperature")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    openai_api_key = args.api_key

    current_script_path = os.path.abspath(__file__)
    MAD_path = current_script_path.rsplit("/", 2)[0]

    config = json.load(open(f"{MAD_path}/code/utils/config.json", "r"))

    # 读取 JSON 数据
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    save_file_dir = args.output_dir
    if not os.path.exists(save_file_dir):
            os.mkdir(save_file_dir)

    for idx, item in enumerate(tqdm(data)):

        sentence = item["text"]
        label = item["label"]
        prompts_path = f"{save_file_dir}/{id}-config.json"
        config['source'] = sentence
        config['label'] = label

        with open(prompts_path, 'w') as file:
            json.dump(config, file, ensure_ascii=False, indent=4)

        debate = Debate(save_file_dir=save_file_dir, num_players=4, openai_api_key=openai_api_key, prompts_path=prompts_path, temperature=0, sleep_time=0)
        debate.save_file_to_json(id)
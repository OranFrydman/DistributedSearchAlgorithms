import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# global vars initializing


UB = 100
LB = 0
DOMAINS = 10
CONST_P = 0.05

def build_all_neighbors(agents, prob_k):
    neighborhoods = []
    agent_count = 1
    for i in range(len(agents)):
        k = agent_count
        while k < (len(agents)):
            new_neighborhood = Neighborhood(agents[i], agents[k])
            neighborhoods.append(new_neighborhood)
            agents[i].neighborhoods.append(new_neighborhood)
            agents[k].neighborhoods.append(new_neighborhood)
            rnd = random.random()
            if rnd < prob_k:
                agents[i].add_neighbor(agents[k])
                agents[k].add_neighbor(agents[i])
                # new_neighborhood.active=True
            k += 1
        agent_count += 1
        k = agent_count
    return neighborhoods

class Agent:
    AGENT_ID = 0
    def __init__(self):
        self.id = Agent.AGENT_ID
        Agent.AGENT_ID += 1
        self.mailbox = MailBox()
        self.neighbors = []
        self.variable = random.randint(0, DOMAINS - 1)
        self.neighborhoods = []
        self.my_penalty = None
        self.last_found = None

    def get_message(self, mail):
        self.mailbox.box.append(mail)

    def send_message_to_all_neighbors(self):
        for agent in self.neighbors:
            my_message = Message(self.variable, self, agent)
            agent.get_message(my_message)

    def change_var(self, var):
        self.variable = var
        self.my_penalty = self.last_found

    def add_neighbor(self, agent):
        self.neighbors.append(agent)
        index = self.identify_neighbor_index(agent)
        self.change_to_active(index)

    def identify_neighbor_index(self, agent):
        if agent.id > self.id:
            return agent.id - 1
        return agent.id

    def change_to_active(self, agent_index):
        self.neighborhoods[agent_index].active = True

    def get_neighbor_matrix(self, agent):
        index = self.identify_neighbor_index(agent)
        return self.neighborhoods[index]

    def init_penalty(self, clean_mail=False):
        total_penalty = 0
        for mail in self.mailbox.box:
            neighborhood = self.get_neighbor_matrix(mail.publisher)
            matrix = neighborhood.get_mat(self)
            cur_penalty = matrix.iloc[self.variable, mail.message]
            total_penalty += cur_penalty
        self.my_penalty = total_penalty
        if clean_mail:
            self.mailbox.clear_box()

    def read_all_messages_find_best_var(self):
        total_penalty = 0
        min_penalty = self.my_penalty
        best_var = self.variable
        for optional_variable in range(DOMAINS):
            if total_penalty < min_penalty:
                for mail in self.mailbox.box:
                    neighborhood = self.get_neighbor_matrix(mail.publisher)
                    matrix = neighborhood.get_mat(self)
                    cur_penalty = matrix.iloc[optional_variable, mail.message]
                    total_penalty += cur_penalty
                if min_penalty > total_penalty:
                    min_penalty = total_penalty
                    best_var = optional_variable
            total_penalty = 0

        self.last_found = min_penalty
        return best_var
class Message:
    def __init__(self, message, publisher, subscriber):
        self.message = message
        self.publisher = publisher
        self.subscriber = subscriber
class MailBox:
    def __init__(self):
        self.box = []
        self.history = []

    def add_mail(self, mail):
        self.box.append(mail)

    def clear_box(self):
        self.history.extend(self.box)
        self.box = []
class Neighborhood:
    def __init__(self, agent1, agent2):
        self.agent1 = agent1
        self.agent2 = agent2
        self.penalty_mat1 = self.build_mat()
        self.penalty_mat2 = self.penalty_mat1.transpose()
        self.active = False

    def build_mat(self):
        matrix = pd.DataFrame(columns=range(0, DOMAINS), index=range(0, DOMAINS))  # create 10x10 matrix
        for i in range(0, 10):
            for j in range(0, 10):
                rand = np.random.randint(LB, UB)  # add random numbers
                matrix.iloc[i, j] = rand
        return matrix

    def get_mat(self, agent):
        if agent == self.agent1:
            return self.penalty_mat1
        return self.penalty_mat2
class AllocationProblem:
    def __init__(self, agents, iterations):
        self.agents = agents
        self.iterations = iterations

    def solve(self):
        pass

    def init_problem(self):
        for agent in self.agents:
            agent.send_message_to_all_neighbors()
        for agent in self.agents:
            agent.init_penalty()

    def distribute_messages(self):
        for agent in self.agents:
            agent.send_message_to_all_neighbors()
class SolverMGM(AllocationProblem):
    def __init__(self, agents, iterations):
        AllocationProblem.__init__(self, agents, iterations)

    def solve(self):
        reduction_dict = {}
        result = []
        score = []
        self.init_problem()
        for i in range(self.iterations):
            if i > 0:
                self.distribute_messages()
            for agent in self.agents:
                best_var = agent.read_all_messages_find_best_var()
                agent_reduction = agent.my_penalty - agent.last_found
                reduction_dict[agent] = [best_var, agent_reduction]
            reduction_dict = dict(
                sorted(reduction_dict.items(), key=lambda x: x[1][1], reverse=False))  # sorting from max to min penalty
            while reduction_dict:
                MGM_change = reduction_dict.popitem()
                MGM_agent = MGM_change[0]
                MGM_var = MGM_change[1][0]
                MGM_agent.change_var(MGM_var)
                reduction_dict = MGM.clean_neighbors_MGM(reduction_dict, MGM_agent)
            for agent in self.agents:
                agent.init_penalty(clean_mail=True)
                result.append(agent.my_penalty)
            score.append(sum(result))
            result = []
        return score
    def clean_neighbors_MGM(self, dict, agent):
        for neighbor in agent.neighbors:
            if neighbor in dict:
                del dict[neighbor]
        return dict
class SolverDSA(AllocationProblem):
    def __init__(self, agents, iterations, p):
        AllocationProblem.__init__(self, agents, iterations)
        self.p = p
    def solve(self):
        result = []  # array list with the penalty of each agent for each round
        score = []  # each iteration its the sum of result and represent the score value for the curr iteration
        self.init_problem()
        for i in range(self.iterations):
            if i > 0:
                self.distribute_messages()
            for agent in self.agents:
                best_var = agent.read_all_messages_find_best_var()
                rand = random.random()
                if rand < self.p:
                    agent.change_var(best_var)
            for agent in self.agents:
                agent.init_penalty(clean_mail=True)
                result.append(agent.my_penalty)
            score.append(sum(result))
            result = []
        return score

# init variables
seed = 10
np.random.seed(seed)
iterations = 50
runs = 1
agents_number = 3
k = 1  # Make Neighbor
p_change_DSA = 0.05
accumulated_score_mgm = [0] * iterations
accumulated_score1 = [0] * iterations
accumulated_score2 = [0] * iterations
accumulated_score3 = [0] * runs
agents = []
P_list = []
score_by_p_list = []
# for i in range(runs):
#     status = (i / runs) * 100
#     if status % 10 == 0:
#         print(f'loading P Graph... {status} %')
#         # initializing the agents
#     for a in range(agents_number):
#         agent = Agent()
#         agents.append(agent)
#     # determine neighbors for each agent - build the matrices
#     all_neighborhoods = build_all_neighbors(agents, k)
#     # Creating the Problems
#     agents_temp = copy.deepcopy(agents)
#     for j in range(runs):
#         if j == i:
#             P_list.append(p_change_DSA)
#         DSA3 = SolverDSA(agents, iterations, p=p_change_DSA)
#         score_by_p_list = DSA3.solve()
#         accumulated_score3[j] = score_by_p_list[-1] + accumulated_score3[j]
#         p_change_DSA += CONST_P
#         agents = copy.deepcopy(agents_temp)
#         # reset variables
#     score3 = []
#     agents = []
#     Agent.AGENT_ID = 0
#     p_change_DSA = 0
# average_score3 = np.array(accumulated_score3) / runs
# df2 = pd.DataFrame({'p': P_list, 'Penalty Score': average_score3})
# b = df2.plot.scatter(x='p', y='Penalty Score')


for i in range(runs):
    status = (i / runs) * 100
    if status % 10 == 0:
        print(f'loading ... {status} %')
    # initializing the agents
    for j in range(agents_number):
        agent = Agent()
        agents.append(agent)
    # determine neighbors for each agent - build the matrices
    all_neighborhoods = build_all_neighbors(agents, k)
    # deep-copy for each solver
    agents2 = copy.deepcopy(agents)
    agents4 = copy.deepcopy(agents)
    agents3 = copy.deepcopy(agents)
    temp_agents = copy.deepcopy(agents)
    # Creating the Problems
    DSA1 = SolverDSA(agents, iterations, p=1)
    DSA2 = SolverDSA(agents2, iterations, p=0.2)
    MGM = SolverMGM(agents4, iterations)
    # solving the problems
    score1 = DSA1.solve()
    score2 = DSA2.solve()
    score_mgm = MGM.solve()
    # Calculating score penalties for each problem
    accumulated_score1 = np.array(score1) + np.array(accumulated_score1)
    accumulated_score2 = np.array(score2) + np.array(accumulated_score2)
    accumulated_score_mgm = np.array(score_mgm) + np.array(accumulated_score_mgm)
    for j in range(runs):
        if j == i:
            P_list.append(p_change_DSA)
        DSA3 = SolverDSA(agents3, iterations, p=p_change_DSA)
        score_by_p_list = DSA3.solve()
        accumulated_score3[j] = score_by_p_list[-1] + accumulated_score3[j]
        p_change_DSA += CONST_P
        agents3 = copy.deepcopy(temp_agents)
        # reset variables
    score3 = []
    agents = []
    Agent.AGENT_ID = 0
    p_change_DSA = 0.05
    # reset variables
    score1 = []
    score2 = []
    score3 = []
    score_mgm = []
    agents = []
    all_neighborhoods = []
    Agent.AGENT_ID = 0

# # Preparing plots
average_score1 = accumulated_score1 / runs
average_score2 = accumulated_score2 / runs
average_score_mgm = accumulated_score_mgm / runs
average_score3 = np.array(accumulated_score3) / runs
dfPlot = pd.DataFrame()
dfPlot['DSA p=0.7'] = pd.Series(average_score1)
dfPlot['DSA p=0.2'] = pd.Series(average_score2)
dfPlot['MGM'] = pd.Series(average_score_mgm)
a = dfPlot.plot.line(title='Penalty Score by Iterations, ')
df2 = pd.DataFrame({'p': P_list, 'Penalty Score': average_score3})
b = df2.plot.scatter(x='p', y='Penalty Score')
plt.show()

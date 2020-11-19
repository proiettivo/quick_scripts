import numpy as np
from tqdm import tqdm
import os, pickle

relu = np.vectorize(lambda x: max(x, 0))

#Count stuff in v in [lb, ub) or [lb, ub]
def count_occ(v, lb, ub, r_include=False):
    g = v >= lb
    s = v <= ub if r_include else v < ub
    return (g & s).sum()


class WeightFactory:

    #Shitty 1-dim input, 1-dim output, simple hidden layers dim
    def __init__(self, input_dim, output_dim, hidden_layers_dims):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hl = hidden_layers_dims

    def get_random_weights(self):
        weights = ([], [])

        sigma = np.sqrt(2/(self.hl[0] + self.input_dim))
        weights[0].append(np.random.normal(0, sigma, (self.hl[0], self.input_dim)))
        weights[1].append(np.zeros((self.hl[0],)))

        if len(self.hl) > 1:
            for i in range(len(self.hl)-1):
                hl_dim = self.hl[i+1]
                prev_hl_dim = self.hl[i]
                sigma = np.sqrt(2 / (hl_dim + prev_hl_dim))
                weights[0].append(np.random.normal(0, sigma, (hl_dim, prev_hl_dim)))
                weights[1].append(np.zeros((hl_dim,)))

        sigma = np.sqrt(2 / (self.output_dim + self.hl[-1]))
        weights[0].append(np.random.normal(0, sigma, (self.output_dim, self.hl[-1])))
        weights[1].append(np.zeros((self.output_dim,)))
        return weights

    @staticmethod
    def get_son(parent_weights_1, parent_weights_2, learning_rate):
        weights = ([], [])
        for k in range(len(parent_weights_1)):
            for i in range(len(parent_weights_1[k])):
                pw = parent_weights_1[k][i]
                pw2 = parent_weights_2[k][i]
                pws = pw.shape
                sigma = np.sqrt(2/sum(pws))
                weights[k].append((pw + pw2)*0.5 + np.random.normal(0, sigma*learning_rate, pws))

        return weights


#Shitty game specific MLP
class GameMlp:

    def __init__(self, weights):
        self.w_weights, self.b_weights = weights
        self.weights = weights

    def get_weights(self):
        return self.weights

    def decide(self, ptl, ptw, own_funds, other_funds, wins_draws):
        input = np.array([ptl, ptw, own_funds, other_funds, wins_draws])
        for i in range(len(self.w_weights[:-1])):
            w = self.w_weights[i]
            b = self.b_weights[i]
            input = relu(np.dot(w, input) + b)

        #linear output
        return np.dot(self.w_weights[-1], input) + self.b_weights[-1]


class Player:

    def __init__(self, is_left, pos, tot_pos, wins_draws):
        self.is_left = is_left
        self.pos = pos
        self.tot_pos = tot_pos
        self.wins_draws = wins_draws
        self.own_funds = 1
        self.other_funds = 1

    def investment(self, pos, own_funds, other_funds):
        self.pos = pos
        self.own_funds = own_funds
        self.other_funds = other_funds

        # Don't waste time learning the boundaries of the game
        return max(min(self.reason(), self.own_funds), 0)

    def reason(self):
        pass


class RandomPlayer(Player):

    def reason(self):
        return np.random.rand()


class WisePlayer(Player):

    def __init__(self, is_left, pos, tot_pos, wins_draws, weights):
        super().__init__(is_left, pos, tot_pos, wins_draws)
        self.brain = GameMlp(weights)

    def reason(self):
        ptl = self.pos-1 if self.is_left else self.tot_pos - self.pos
        ptw = self.pos-1 if not self.is_left else self.tot_pos - self.pos
        return self.brain.decide(ptl, ptw, self.own_funds, self.other_funds, self.wins_draws)


class GameOfStrangeRisk:

    def __init__(self, pos, tot_pos, left_player, right_player, verbose):
        self.pos = pos
        self.tot_pos = tot_pos
        self.left_player = left_player
        self.right_player = right_player
        self.lpf = 1
        self.rpf = 1
        self.v = verbose
        assert left_player.is_left == 1, "LEFT PLAYER IS NOT LEFT"
        assert not (right_player.is_left == 1), "RIGHT PLAYER IS NOT RIGHT"
        assert not ((pos < 1) or (pos > tot_pos)), "STARTING POS WRONG"

    def play_round(self):
        lw = self.left_player.investment(self.pos, self.lpf, self.rpf)
        rw = self.right_player.investment(self.pos, self.rpf, self.lpf)
        left_win = lw > rw

        if lw == rw:
            if self.left_player.wins_draws == self.right_player.wins_draws:
                if self.v:
                    print("Left waged: {}, Right Waged {}, Funds L {}, Funds R {}, Pos {}".format(lw, rw, self.lpf,
                                                                                              self.rpf, self.pos))
                return
            elif self.left_player.wins_draws == 1:
                left_win = True
            else:
                left_win = False

        if left_win:
            self.pos = self.pos + 1
            self.lpf = self.lpf - lw
        else:
            self.pos = self.pos - 1
            self.rpf = self.rpf - rw

        if self.v:
            print("Left waged: {}, Right Waged {}, Funds L {}, Funds R {}, Pos {}".format(lw, rw, self.lpf,
                                                                                      self.rpf, self.pos))

        if self.pos <= 1:
            if self.v:
                print("THE END Right wins")
            return [0, 3]
        elif self.pos >= self.tot_pos:
            if self.v:
                print("THE END Left wins")
            return [3, 0]

        if (self.lpf == 0) and (self.lpf == self.rpf):
            return [1, 1]

    def play_game(self, length=100):
        for _n in range(1, length):
            res = self.play_round()
            if res is not None:
                return res

            if _n == 99:
                return [1, 1]


if __name__ == "__main__":

    start_pos = 4
    tot_pos = 7
    l_wins_draws = 0
    r_wins_draws = 0
    learning_rate = 0.01
    verb = False
    test_vs_random = 100

    print("Running a very stupid game...")

    gen_size = 100
    new_comers = 5

    max_generations = 100

    wf = WeightFactory(5, 1, [32, 32])

    generation = [wf.get_random_weights() for i in range(gen_size)]

    for gen in range(max_generations):
        points = np.zeros(gen_size)
        first_left_move = np.zeros(gen_size)
        points_vs_random = np.zeros(gen_size)
        for i in tqdm(range(gen_size)):
            lp = WisePlayer(1, start_pos, tot_pos, l_wins_draws, generation[i])
            first_left_move[i] = lp.investment(start_pos,1,1)
            for _ in range(test_vs_random):
                lp = WisePlayer(1, start_pos, tot_pos, l_wins_draws, generation[i])
                rlp = RandomPlayer(1, start_pos, tot_pos, l_wins_draws)
                rp = WisePlayer(0, start_pos, tot_pos, r_wins_draws, generation[i])
                rrp = RandomPlayer(0, start_pos, tot_pos, r_wins_draws)
                rs_1 = GameOfStrangeRisk(start_pos, tot_pos, lp, rrp, verb).play_game()
                rs_2 = GameOfStrangeRisk(start_pos, tot_pos, rlp, rp, verb).play_game()
                points_vs_random[i] = points_vs_random[i] + rs_1[0] + rs_2[1]

            for k in [pos for pos in range(gen_size) if pos != i]:
                lp = WisePlayer(1, start_pos, tot_pos, l_wins_draws, generation[i])
                rp = WisePlayer(0, start_pos, tot_pos, r_wins_draws, generation[k])
                if verb:
                    print("Playing {} vs {}".format(i, k))
                game = GameOfStrangeRisk(start_pos, tot_pos, lp, rp, verb)
                res = game.play_game()
                points[i] = points[i] + res[0]
                points[k] = points[k] + res[1]

        points_2_prob = (points + points_vs_random) / (points_vs_random + points.sum())
        pos_best = points_2_prob.argmax()

        print("Generation {}, best nr: {}, 1st move: {}, avg pts: {}, avg vs rnd: {}".format(gen, pos_best,
                                                                    first_left_move[pos_best],
                                                                    points[pos_best] / (2 * (gen_size - 1)),
                                                                    points_vs_random[pos_best]/ (2 * test_vs_random)))

        avg_points = points / (2 * (gen_size - 1))
        points_stats = (avg_points.mean(), count_occ(avg_points, 0, 0.5),
                        count_occ(avg_points, 0.5, 1),
                        count_occ(avg_points, 1, 1.5),
                        count_occ(avg_points, 1.5, 2),
                        count_occ(avg_points, 2, 2.5),
                        count_occ(avg_points, 2.5, 3, True))
        print("Avg Points: {}! Distr: [0:0.5): {}, [0.5, 1): {}, "
              "[1, 1.5): {}, [1.5, 2): {}, [2, 2.5): {}, [2.5, 3]: {}".format(*points_stats))

        if gen == max_generations-1:
            save_path = os.path.join(os.getcwd(), "best.p")
            pickle.dump(generation[pos_best], open(save_path, "wb"))
        else:
            cdf = points_2_prob.cumsum()
            pos = np.array(range(gen_size))
            new_generation = []
            for i in range(gen_size-new_comers):
                p1_pos = pos[np.random.rand() < cdf].min()
                p2_pos = pos[np.random.rand() < cdf].min()
                new_generation.append(wf.get_son(generation[p1_pos], generation[p2_pos], learning_rate))

            for i in range(new_comers):
                new_generation.append(wf.get_random_weights())

            generation = new_generation




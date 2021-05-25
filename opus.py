from model.graph_mcmc import Graph_MCMC
from inference.softmax import SoftmaxNeuralNet
from utils.subsampling import random_index_arr
import numpy as np
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

polbooks_args = {
    "B": 3,
    "f": 0.7,
    "Tb": 1000,
    "Tt": 10000,
    "s": 0.05,
    "burn-in": 0.4,
    "thinning": 10,
    "k": 1,
    "D'": 3,
    "burn-in-red": 0.4,
    "thinning-red": 10,
    "s-red": 0.05
}

school_args = {
    "B": 10,
    "f": 0.7,
    "Tb": 1000,
    "Tt": 10000,
    "s": 0.2,
    "burn-in": 0.4,
    "thinning": 10,
    "k": 1,
    "D'": 10,
    "burn-in-red": 0.4,
    "thinning-red": 10,
    "s-red": 0.2
}

fb_args = {
    "B": 10,
    "f": 0.7,
    "Tb": 1000,
    "Tt": 10000,
    "s": 0.017,
    "burn-in": 0.4,
    "thinning": 10,
    "k": 1,
    "D'": 10,
    "burn-in-red": 0.4,
    "thinning-red": 10,
    "s-red": 0.5
}




def create_polbooks_graph():
    graph = Graph_MCMC()
    graph.read_from_ns("polbooks")
    graph.rename_property("label", "_label")
    graph.convert_props_to_flags()
    return graph, polbooks_args


def create_school_graph():
    graph = Graph_MCMC()
    graph.read_from_ns("sp_primary_school/day_2")
    graph.rename_property("label", "_label")
    graph.rename_property("viz", "_viz")

    graph.convert_props_to_flags()
    graph.remove_property("Unknown")
    return graph, school_args


def create_maier_graph():
    graph = Graph_MCMC()
    graph.read_from_ns("facebook_friends")
    graph.convert_props_to_flags()
    graph.add_ego_node()
    graph.filter_out_low_degree(min_degree=2)
    args = {
        "B": 18,
        "f": 0.7,
        "Tb": 1000,
        "Tt": 10000,
        "s": 0.2,
        "burn-in": 0.4,
        "thinning": 10,
        "k": 1,
        "D'": 18,
        "burn-in-red": 0.4,
        "thinning-red": 10,
        "s-red": 0.5
    }

    return graph, args



def create_fb_graph():
    
    class Graph_Custom (Graph_MCMC):

        def get_feature_names(self):
            feature_name_map = self.G.graph_properties["feat_names"]
            names = []

            for i in range(0, len(feature_name_map)):
                name = feature_name_map[i]
                name = name.replace(";", "-")
                name = name.replace("anonymized feature ", "")
                names.append(name)
            
            return names


        def generate_feature_matrix(self):
            feat_map = self.G.vertex_properties["feat"]
            feat_names = self.get_feature_names()

            vertices = self.G.get_vertices()
            
            N = len(vertices)
            D = len(feat_names)
            X = np.empty((N, D))

            for vertex_index, vertex_id in enumerate(vertices):
                boolean_vector = feat_map[vertex_id]
                for prop_index in range(0, D):
                    X[vertex_index, prop_index] = boolean_vector[prop_index]
            
            return X

    graph = Graph_Custom()
    graph.read_from_ns("ego_social/facebook_1912")
    return graph, fb_args

def latex_print(means, std_devs, dp=3):
    result = ""
    for i in range(0, len(means)):
        result += "$"
        result += str(round(means[i], dp))
        result += " \\pm "
        result += str(round(std_devs[i], dp))
        result += "$ & "

    print(result)



def run(verbose=False):

    graph, args = create_polbooks_graph()
    #graph, args = create_school_graph()
    #graph, args = create_fb_graph()
    #graph, args = create_maier_graph()

    graph.print_info()
    graph.partition(B_min=args["B"], B_max=args["B"])

    av_dl = graph.mcmc(num_iter=args["Tb"])

    X = graph.generate_feature_matrix()
    Y = graph.generate_posterior()

    D = X.shape[1]
    B = Y.shape[1]

    N = X.shape[0]
    assert N == Y.shape[0]


    train_indices, test_indices = random_index_arr(N, fraction=args["f"])
    X_train, Y_train = X[train_indices, :], Y[train_indices, :]
    X_test, Y_test = X[test_indices, :], Y[test_indices, :]

    classifier = SoftmaxNeuralNet(layers_size=[D, B])
    classifier.perform_mala(X_train, Y_train, step_scaling=args["s"], num_iter=args["Tt"], verbose=verbose)

    classifier.thin_samples(burn_in=args["burn-in"], thin_factor=args["thinning"])

    training_loss = classifier.average_loss_per_point(X_train, Y_train, include_prior=False)
    test_loss = classifier.average_loss_per_point(X_test, Y_test, include_prior=False)

    classifier.compute_mean_variances()
    kept_features, c_star = classifier.gen_top_feature_indices(std_dev_multiplier=args["k"], D_reduced=args["D'"])

    reduced_X_train, reduced_X_test = X_train[:, kept_features], X_test[:, kept_features]

    reduced_D = reduced_X_train.shape[1]
    B = Y_train.shape[1]

    # now train new classifier
    reduced_classifier = SoftmaxNeuralNet(layers_size=[reduced_D, B])
    reduced_classifier.perform_mala(reduced_X_train, Y_train, step_scaling=args["s-red"], num_iter=args["Tt"], verbose=verbose)

    reduced_classifier.thin_samples(burn_in=args["burn-in-red"], thin_factor=args["thinning-red"])

    reduced_training_loss = reduced_classifier.average_loss_per_point(reduced_X_train, Y_train, include_prior=False)
    reduced_test_loss = reduced_classifier.average_loss_per_point(reduced_X_test, Y_test, include_prior=False)

    results = [av_dl, training_loss, test_loss, c_star, reduced_training_loss, reduced_test_loss]
    print("\n~~~~~~~~~~~~~ RESULTS ~~~~~~~~~~~~~~~\n")
    print("S_b, L_0, L_1, c*, L_0', L_1'")
    print(results)

    if verbose:
        graph.draw(gen_layout=False, size=7)
        classifier.plot_U()
        reduced_classifier.plot_U()

    train_loss_arr = classifier.loss_per_class(X_train, Y_train)
    test_loss_arr = classifier.loss_per_class(X_test, Y_test)
    print("\n Losses per class \n")
    print(train_loss_arr)
    print(test_loss_arr)

    train_accuracy = classifier.accuracy_per_class(X_train, Y_train)
    test_accuracy = classifier.accuracy_per_class(X_test, Y_test)
    print("\n Accuracy per class \n")
    print(train_accuracy)
    print(test_accuracy)

    return results


if __name__ == "__main__":
    results_arr = []
    num_iter = 10
    for i in range(0, num_iter):
        print("~~~~~~~~~~~~~~ ITERATION {}/{} ~~~~~~~~~~~~~~~".format(i+1, num_iter))
        res = run()
        results_arr.append(res)

    means = np.mean(results_arr, axis=0)
    std = np.std(results_arr, axis=0)
    print("\n~~~~~~~~~~~~~ FINISH ~~~~~~~~~~~~~~~\n")
    print("S_b, L_0, L_1, c*, L_0', L_1'")
    print(means)
    print(std)
    latex_print(means, std)

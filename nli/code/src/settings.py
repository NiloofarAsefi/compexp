"""Module containing the class for the settings.
It is an adaptation of the code referenced in:
https://github.com/jayelm/compexp/blob/master/vision/settings.py"""

# Change CCE_vision class setting to CCE_nli setting
# class Settings:
#     """
#     Class that stores all the settings used in each run.
#     """

#     def __init__(
#         self,
#         *,
#         subset,
#         model,
#         pretrained,
#         beam_limit,
#         device,
#         dataset="places365",
#         root_models="data/model/",
#         root_datasets="data/dataset/",
#         root_segmentations="data/cache/segmentations/",
#         root_activations="data/cache/activations/",
#         root_results="data/results/",
#     ):
#         self.index_subset = subset
#         self.model = model
#         self.dataset = dataset
#         if pretrained == "places365" or pretrained == "imagenet":
#             self.pretrained = pretrained
#         else:
#             self.pretrained = None
#         self.num_clusters = num_clusters
#         self.beam_limit = beam_limit
#         self.dir_datasets = root_datasets
#         self.__root_segmentations = root_segmentations
#         self.__root_activations = root_activations
#         self.__root_results = root_results
#         self.__root_models = root_models
#         self.device = device

    


#     def get_num_classes(self):
#         """
#         Returns the number of classes of the dataset.
#         """
#         if self.dataset == "places365":
#             return 365
#         elif self.dataset == "imagenet":
#             return 1000
#         elif self.dataset == "ade20k":
#             return 365

#     def get_model_file_path(self):
#         """
#         Returns the path to the pretrained weights of the model.
#         """
#         if self.pretrained == "places365":
#             if self.model == "densenet161":
#                 model_file_name = (
#                     "whole_densenet161_places365_python36.pth.tar"
#                 )
#             else:
#                 model_file_name = f"{self.model}_places365.pth.tar"
#             return self.__root_models + "zoo/" + model_file_name
#         elif self.pretrained == "imagenet":
#             return None
#         else:
#             return "<UNTRAINED>"

#     def get_weights(self):
#         """
#         Returns the pretrained weights of the model.
#         """
#         if self.pretrained == "imagenet":
#             return "IMAGENET1K_V1"
#         elif self.pretrained == "places365":
#             return self.get_model_file_path()
#         else:
#             return None

#     def get_data_directory(self):
#         """
#         Returns the directory where the data is stored.
#         """
#         if self.model != "alexnet":
#             return f"{self.dir_datasets}broden1_224"
#         else:
#             return f"{self.dir_datasets}broden1_227"

#     def get_model_root(self):
#         """
#         Returns the root directory where the models are stored.
#         """
#         return self.__root_models


#     def get_parallel(self):
#         """
#         Returns True if the model is parallelized.
#         """
#         if self.dataset == "places365":
#             return True
#         else:
#             return False

    

#     def get_activation_directory(self):
#         """
#         Returns the directory where the activations are stored.
#         """
#         return (
#             f"{self.__root_activations}{self.model}/"
#             f"{self.index_subset}/"
#             f"{'w_'+self.pretrained if self.pretrained else 'untrained'}"
#         )

#     def get_results_directory(self):
#         """
#         Returns the directory where the results are stored.
#         """
#         return (
#             f"{self.__root_results}{self.dataset}/{self.index_subset}/"
#             f"{'w_'+self.pretrained if self.pretrained else 'untrained'}/"
#             f"{self.model}"
#         )

#     def get_info_directory(self):
#         """
#         Returns the directory where the info files are stored.
#         """
#         return (
#             f"{self.__root_segmentations}{self.dataset}/"
#             f"{self.index_subset}/{self.get_img_size()}"
#         )


import os

class Settings:
    """
    Class that stores all settings for clustered compositional explanations (CCE) in NLI tasks.
    """
    def __init__(
        self,
        subset,
        model="models/bowman_snli/6.pth",
        model_type="bowman",
        root_models="models/",  # Base directory for models, I already start from compexp/nli
        pretrained=None,
        num_clusters=5,   #change to 5
        beam_limit=10,
        device="cpu",
        dataset="snli",
        root_datasets="data/dataset/",
        root_results="data/results/",
        metric="iou",
        max_formula_length=5,
        complexity_penalty=1.00,
        parallel=4,
        random_weights=False,
        n_sentence_feats=2000,
        data_file="data/analysis/snli_1.0_dev.feats",
        shuffle=False,  # Add shuffle with a default value
        debug=False,  # Add debug with a default value
        feat_type="sentence"  # Default to "sentence" if unspecified
    ):  
        # Core settings
        self.index_subset = subset
        self.model = model
        self.model_type = model_type
        self.dataset = dataset
        self.pretrained = pretrained if pretrained in ["snli", "mnli"] else None
        self.num_clusters = num_clusters
        self.beam_limit = beam_limit
        self.device = device
        self.metric = metric
        self.max_formula_length = max_formula_length
        self.complexity_penalty = complexity_penalty
        self.parallel = parallel
        self.random_weights = random_weights
        self.n_sentence_feats = n_sentence_feats
        self.data_file = data_file
        self.shuffle = shuffle
        self.debug = debug
        self.feat_type = feat_type
        
        # Root directories
        self.dir_datasets = root_datasets
        self.__root_results = root_results
        self.__root_models = root_models

        # Derived settings
        self.vecpath = self._get_vecpath()
        self.result_path = self._construct_result_path()

    def _get_vecpath(self):
        """
        Returns the path to the vector file based on the data file path.
        """
        assert self.data_file.endswith(".feats"), "Data file should have a .feats extension."
        return self.data_file.replace(".feats", ".vec")

    def get_num_classes(self):
        """
        Returns the number of classes for the NLI dataset.
        """
        return 3  # entailment, neutral, contradiction

    def get_model_file_path(self):
        """
        Returns the path to the pretrained model file for NLI.
        """
        # Use the model directly without appending dataset or additional extensions
        return f"{self.__root_models}{self.model}"

    def get_results_directory(self):
        """
        Returns the directory where the clustering results are stored.
        """
        return (
            f"{self.__root_results}{self.dataset}/{self.index_subset}/"
            f"{self.model}-clusters{self.num_clusters}/"
        )
    
    def get_mask_shape(self):
        """
        Returns the shape of the mask based on the model type specified in settings.
        For the Bowman model in NLI tasks, it returns a sentence-level embedding shape.
        
        Returns:
        - Tuple[int]: Shape of the mask as (mlp_linear_output,1) for Bowman model in NLI.
        - Defaults to vision model shapes if model type is not Bowman.
        """
        if self.model_type == "bowman":
            # Assuming Bowman model in NLI task uses sentence-level embeddings
            mlp_linear_output = 1024  # Replace with actual encoder dimension if different
            return (mlp_linear_output,1)
        else:
            # Default mask shapes for vision models in CCE
            return (112, 112) if self.model_type != "alexnet" else (113, 113)

    def get_max_mask_size(self):
        """
        Returns the maximum size of the mask based on the model type.

        For vision models, it calculates the area (width * height).
        For NLI models, it returns the embedding dimension size (sentence-level mask).

        Returns:
        - int: Maximum mask size based on model type.
        """
        mask_shape = self.get_mask_shape()
        
        if len(mask_shape) == 2:   # Vision model case
            return mask_shape[0] * mask_shape[1]
        else:                     # NLI model case, like Bowman
            return mask_shape[0]

    def get_model_root(self):
        """
        Returns the root directory where the models are stored.
        """
        return self.__root_models

    def _construct_result_path(self):
        """
        Constructs and returns the path to save experiment results based on settings.
        """
        mbase = os.path.splitext(os.path.basename(self.model))[0]
        dbase = os.path.splitext(os.path.basename(self.data_file))[0]
        return (
            f"exp/{dbase}-{mbase}-sentence-{self.max_formula_length}"
            f"{'-shuffled' if self.shuffle else ''}"
            f"{'-debug' if self.debug else ''}"
            f"{f'-{self.metric}' if self.metric != 'iou' else ''}"
            f"{'-random-weights' if self.random_weights else ''}"
        )

    def get_clustering_settings(self):
        """
        Returns clustering-specific settings as a dictionary for easy access.
        """
        return {
            "num_clusters": self.num_clusters,
            "metric": self.metric,
            "max_formula_length": self.max_formula_length,
            "complexity_penalty": self.complexity_penalty,
        }

    def print_settings(self):
        """
        Prints all current settings for verification.
        """
        print("Current NLI CCE Settings:")
        print(f"Subset: {self.index_subset}")
        print(f"Model: {self.model}")
        print(f"Pretrained: {self.pretrained}")
        print(f"Number of Clusters: {self.num_clusters}")
        print(f"Beam Limit: {self.beam_limit}")
        print(f"Device: {self.device}")
        print(f"Dataset: {self.dataset}")
        print(f"Metric: {self.metric}")
        print(f"Max Formula Length: {self.max_formula_length}")
        print(f"Complexity Penalty: {self.complexity_penalty}")
        print(f"Parallel Processing: {self.parallel}")
        print(f"Random Weights: {self.random_weights}")
        print(f"Sentence Features: {self.n_sentence_feats}")
        print(f"Data File: {self.data_file}")
        print(f"Vector Path: {self.vecpath}")
        print(f"Results Directory: {self.get_results_directory()}")
        
    def get_info_directory(self):
        """
        Returns the directory where the info files for NLI tasks are stored.
        """
        # Customize the base directory and structure for NLI
        return (
            f"{self.__root_results}{self.dataset}/{self.index_subset}/"
            f"{self.model}-info/"
        )
    
    
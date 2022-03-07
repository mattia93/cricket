from tensorflow.keras.utils import Sequence
import numpy as np
import random
from dataclasses import dataclass
import pickle


@dataclass
class SimplePlanGenerator(Sequence):
    filenames: list
    actions_dict: dict
    max_dim: int
    batch_size: int = 64
    shuffle: bool = True

    def __getitem__(self, index):
        batch = self.filenames[index * self.batch_size : (index + 1) * self.batch_size]
        X = np.zeros((self.batch_size, self.max_dim))
        Y = np.zeros((self.batch_size, self.max_dim))
        for i, filename in enumerate(batch):
            lines = load_file(filename)
            plan, y = parse_observations(lines, self.actions_dict)
            for pos, val in enumerate(plan):
                if pos < self.max_dim:
                    X[i][pos] = val
                    Y[i][pos] = y[pos]
                else:
                    break
        return X, Y

    def __len__(self):
        return len(self.filenames) // self.batch_size

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle == True:
            np.random.shuffle(self.filenames)


def load_file(file: str, binary: bool = False, use_pickle: bool = False):
    '''
    Get file content from path.

    Args:
        file:
            A string that contains the path
            to the file.
        binary:
            Optional. True if the file is a
            binary file.
        use_pickle:
            Optional. True if the file was
            saved using pickle.

    Returns:
        The content of the file.

    Raises:
        FileNotFoundError:
            An error accessing the file
    '''
    operation = 'r'
    if binary:
        operation += 'b'
    with open(file, operation) as rf:
        if use_pickle:
            output = pickle.load(rf)
        else:
            output = rf.readlines()
        rf.close()
    return output


def retrieve_from_dict(key: str, dictionary: dict):
    '''
    Return the dictionary value given the key.

    Args:
        key:
            A string that is the key.
        dictionary:
            A dict.

    Returns:
        The value corresponding to the key.

    Raises:
        KeyError:
            An error accessing the dictionary.
    '''

    msg_error = f'Key {key.upper()} is not in the dictionary'

    try:
        return dictionary[key.upper()]
    except KeyError:
        print(msg_error)
        np.random.seed(47)
        return np.random.randint(0, len(dictionary))


def remove_parentheses(line: str) -> list:
    '''
    Remove parentheses from a string.

    Args:
        line: a string that is enclosed in parentheses.
        For example:

        "(string example)"

    Returns:
        The string without the parenteses.
        None if the string is empty.

    Raises:
        FileFormatError: error handling the string
    '''

    msg = (f'Error while parsing a line. Expected "(custom '
           + f'text)" but found "{line}"')

    line = line.strip()
    if line.startswith('(') and ')' in line:
        element = line[1:]
        input_el = element.rsplit(')', 1)[0]
        input_el = input_el.strip()
        o = element.rsplit(')', 1)[1]
        if o.strip() == 'O':
            o = False
        elif 'NOISE' in o:
            o = True
        else:
            return None
        if len(input_el) == 0:
            return None
        else:
            return input_el, o
    elif len(line) == 0:
        return None
    else:
        raise FileFormatError(msg)


def parse_observations(lines: list, obs_dict: dict = None) -> list:
    '''
    Removes parentheses and empty strings from
    the observations list.

    Args:
        lines:
            List of strings that contains the
            observations. Each observation is
            enclosed in parentheses. For
            example:

            ['(observation1)', '', '(observation2)']

        obs_dict:
            Optional. A dictionary that maps each
            observation to its unique identifier.

    Returns:
        The input list without parentheses and
        empty strings.

    Raises:
        FileFormatError:
            An error accessing the file.
    '''
    msg_empty = 'Observations list is empty.'

    observations = list()
    y_true_list = list()

    for line in lines:
        observation, y_true = remove_parentheses(line)
        if observation is not None:
            if obs_dict is not None:
                observation = retrieve_from_dict(observation, obs_dict)
            observations.append(observation)
            y_true_list.append(y_true)
    if len(observations) > 0:
        return observations, y_true_list
    else:
        raise FileFormatError(msg_empty)


class FileFormatError(Exception):
    pass


np.random.seed(43)
random.seed(43)

#!/usr/bin/env python3

def _modify_line(line, config):
    output = line
    if config["Text"]["tolowercase"]:
        output = output.lower()

    if config["Text"]["firstuppercase"] > 0.0:
        prob = config["Text"]["firstuppercase"]
        words = output.split()
        output = ""
        for word in words:
            if np.random.random() < prob:
                output += word.title()
            else:
                output += word

            output += " "

    if config["Text"]["punctuationafterword"]:
        prob = config["Text"]["punctuationafterword"]
        words = output.split()
        output = ""
        punctuations = config["Text"]["punctuations"]
        for word in words:
            output += word

            if np.random.random() < prob:
                output += punctuations[np.random.randint(len(punctuations))]

            output += " "

    return output.rstrip()


def modify_content(content, config):
    new_content = []
    for line in content:
        new_content.append(_modify_line(line, config))

    return new_content


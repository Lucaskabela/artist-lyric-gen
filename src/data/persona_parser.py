"""
persona_parser.py

PURPOSE: This file defines and creates persona objects from a csv or json
and handles exporting them.
"""

import argparse
import csv
import json


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for persona generation",
    )
    parser.add_argument(
        "--persona-data",
        type=str,
        default="./personas.csv",
        help="File containing personas",
    )
    parser.add_argument(
        "--save-json-data",
        type=str,
        default="./personas.json",
        help="File to save personas",
    )
    parser.add_argument(
        "--save-persona-txt",
        type=str,
        default="./personas.txt",
        help="Text file to save natural personas",
    )
    parser.add_argument("--save-persona-json", action="store_true", default=False)
    parser.add_argument("--save-persona", action="store_true", default=False)
    parser.add_argument("--natural", action="store_true", default=False)
    args = parser.parse_args()
    return args


class PersonaEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Persona):
            return obj.to_json()
        return json.JSONEncoder.default(self, obj)


class Persona:
    def __init__(self):
        self.name = None

    def to_json(self):
        return json.dumps(self.__dict__)

    @staticmethod
    def from_json(bio):
        persona = Persona()
        persona.__dict__ = json.loads(bio)
        return persona

    @staticmethod
    def from_csv_line(line):
        persona = Persona()
        persona.name = line[0]
        persona.real_name = line[1]
        persona.city = line[2]
        persona.nicknames = line[3]
        persona.year = line[4]
        persona.group = line[5]
        persona.discog = line[6]
        return persona

    def to_nn_input(
        self,
        use_rn=True,
        use_city=True,
        use_nn=True,
        use_group=True,
        use_discog=True,
        use_year=True,
    ):
        res = ["<name>", self.name]
        if use_rn and not self.real_name == "":
            res.append("<realname>")
            res.append(self.real_name)
        if use_city and not self.city == "":
            res.append("<city>")
            res.append(self.city)
        if use_nn and not self.nicknames == "":
            res.append("<nicknames>")
            res.append(self.nicknames)
        if use_group and not self.group == "":
            res.append("<groups>")
            res.append(self.group)
        if use_discog and not self.discog == "":
            res.append("<albums>")
            res.append(self.discog)
        if use_year and not self.year == "":
            res.append("<year>")
            res.append(self.year)
        return " ".join(res)

    def to_natural_input(
        self,
        use_rn=True,
        use_city=True,
        use_nn=True,
        use_group=True,
        use_discog=True,
        use_year=True,
    ):
        res = ["I am {}.".format(self.name)]
        if use_rn and not self.real_name == "":
            res.append("My real name is {}.".format(self.real_name))
        if use_city and not self.city == "":
            res.append("I come from {}.".format(self.city))
        if use_nn and not self.nicknames == "":
            res.append("I am also known as {}.".format(self.nicknames))
        if use_group and not self.group == "":
            res.append("I have been a part of groups like {}.".format(self.group))
        if use_discog and not self.discog == "":
            res.append("I have released albums such as {}.".format(self.discog))
        if use_year and not self.year == "":
            res.append("I have been rapping since {}.".format(self.year))
        return " ".join(res)

def create_personas(persona_file_name):
    """
    Creates a dictionary of personas for use in data processing
    Args:
        persona_file_name: the file to load personas from .json or .csv
    Returns:
        personas: Dict[str, Persona] which maps artist names to personas
    """
    personas = {}
    with open(persona_file_name, 'r') as persona_file:

        if persona_file_name.endswith("csv"):
            csv_reader = csv.reader(persona_file, delimiter=",")
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    persona = Persona.from_csv_line(row)
                    personas[persona.name] = persona
                line_count += 1
        else:
            personas = json.load(persona_file)
            for k, v in personas.items():
                personas[k] = Persona.from_json(v)
    return personas


def save_personas(save_file_name, personas):
    with open(save_file_name, 'w') as save_file:
        json.dump(personas, save_file, cls=PersonaEncoder)

def save_personas_txt(save_file_name, personas, natural=False):
    with open(save_file_name, 'w') as save_file:
        for persona in personas.values():
            written_descrip = persona.to_nn_input()
            if natural:
                written_descrip = persona.to_natural_input()
            save_file.write("".join([written_descrip, "\n"]))

def main():
    args = _parse_args()
    personas = create_personas(args.persona_data)
    if args.save_persona_json:
        save_personas(args.save_json_data, personas)
    if args.save_persona:
        save_personas_txt(args.save_persona_txt, personas, args.natural)
    return personas


if __name__ == "__main__":
    main()

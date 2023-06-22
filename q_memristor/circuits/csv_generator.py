import csv

class csv_gen:
    def __init__(self, filename):
        self.filename = filename

    def write_data(self, *arrays):
        data = zip(*arrays)
        with open(self.filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        print(f"Data successfully written to {self.filename}.")

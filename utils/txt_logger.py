class txt_logger():
    def __init__(self, path):
        self.file_path = path
        self.content = []

    def add_info(self, info):
        self.content.append(info)

    def output(self):
        with open(self.file_path, 'w') as f:
            for info in self.content:
                f.write(info + '\n')

    def output_tail(self):
        with open(self.file_path, 'a') as f:
            for info in self.content:
                f.write(info + '\n')

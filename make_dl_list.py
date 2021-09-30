
import directories


START = "CA"
END = "D5"

def main():
    nums = range(int(START, base=36), int(END, base=36)+1)
    sub_dirs = [directories.base36encode(n, 2) for n in nums]

    for sub_dir in sub_dirs:
        yield f"http://192.168.1.104:8080/v7/pics/{sub_dir}/current_state.pickle"

    for sub_dir in sub_dirs:    
        yield f"http://192.168.1.104:8080/v7/pics/{sub_dir}/history.db"

    for sub_dir in sub_dirs:
        yield f"http://192.168.1.104:8080/v7/pics/{sub_dir}/{sub_dir}-x264.mp4"

if __name__ == "__main__":
    for x in main():
        print(x)


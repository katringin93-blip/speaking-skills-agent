import time

def main():
    print("SpeakingSkillsAgent: started")
    print("This is a test build. Next step: OBS folder watcher.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("SpeakingSkillsAgent: stopping")

if __name__ == "__main__":
    main()

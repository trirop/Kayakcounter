import os
# Function to rename multiple files
def main():
   i = 252
   path="/home/tristan/Bilder/Temp2/"
   for filename in os.listdir(path):
      print(filename)
      if "JPG" in filename:
        print(filename)
        my_dest =str(i) + ".jpg"
        my_source =path + filename
        my_dest =path + my_dest
        os.rename(my_source, my_dest)
        i += 1
# Driver Code
if __name__ == '__main__':
   # Calling main() function
   main()

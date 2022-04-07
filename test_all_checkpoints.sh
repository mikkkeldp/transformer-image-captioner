
# choose checkpoints to test
<<<<<<< HEAD
for i in {8..10};
=======
for i in {6..11};
>>>>>>> multi-level
do
   : 
    echo "CHECKPOINT $i" 
    python3 test.py --checkpoint ./checkpoint_"$i".pth
done



# choose checkpoints to test
for i in {4..10};
do
   : 
    echo "CHECKPOINT $i" 
    python3 test.py --checkpoint ./checkpoint_"$i".pth
done

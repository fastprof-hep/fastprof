#grep @ $1 > grp1
#grep @ $2 > grp2
diff $1 $2
exit $?


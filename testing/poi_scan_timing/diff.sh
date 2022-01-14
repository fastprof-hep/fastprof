grep " time" $1
grep @ $1 > grp1
grep @ $2 > grp2
diff grp1 grp2
exit $?


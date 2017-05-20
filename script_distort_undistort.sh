for inp in a; do
	for cage in 1 2 3 ; do
		bitbucket/ocr/Code/Build/Output/bin/RenderPage -c r$cage.txt $inp.png dr$cage.png 2592 4608
		bitbucket/ocr/Code/Build/Output/bin/RenderPage -i -c g$cage.txt dr$cage.png dr$cage_g$cage.png 2550 3300
	done
done

import SimpleHTTPServer
import SocketServer
import os

PORT = 80

web_dir = "../results"
os.chdir(web_dir)

Handler = SimpleHTTPServer.SimpleHTTPRequestHandler

httpd = SocketServer.TCPServer(("", PORT), Handler)

print "serving at port", PORT
print "http://localhost:"+str(PORT)
httpd.serve_forever()
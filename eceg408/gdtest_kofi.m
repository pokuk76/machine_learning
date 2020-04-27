% gradient descent classification example

% define the number of samples per set
NX = 200; % number of blue data points
NY = 200; % number of red data points

% generate some random red and blue data points
VX = 1.7; VY = 1.7;
X=[4;5]+VX*randn(2,round(NX/2));
X=cat(2,X,[7;2]+randn(2,round(NX/2)));
Y=[2;1]+VY*randn(2,NY);

% initialize vector w and scalar b randomly
w=randn(2,1); w=w/norm(w); b=randn(1);

% compute the gradient factors alpha and beta
aa=sum(X,2)-sum(Y,2);
alpha1=aa(1);
alpha2=aa(2);
beta=size(X,2)-size(Y,2);

for n=1:150

    % ------------------
    % Update the w-Vector
    % ------------------
    
    % compute the w-gradient
    dw = zeros(size(w));
    for m=1:size(w,1)
        dw(m)=aa(m)/sqrt(sum(w)) - (w(m)*(aa.'*w))/((w.'*w).^(3/2));
    end
    % edit here and return in dw the current gradient vecor of w

    % update the w vector with a learning rate of 0.1
    w=w+0.1*dw;
    % renormalize the w vector
    w=w/norm(w);
    
    % ------------------
    % Update the b-Value
    % ------------------
    
    % how many X are on the wrong side
    px=w.'*X-b; 
    % how many Y are on the wrong side 
    py=w.'*Y-b;
    % find a mean change based on the negative scores
    db=mean([py(py>0),px(px<0)]);
    % update the scalar value of b
    b=b+db;
    
    % ----------------
    % Draw the Results
    % ----------------
    
    % define the line drawing points
    wx=[w(2);-w(1)];
    A=b*w-20*wx; B=b*w+20*wx;
    C=A+20*w; D=B+20*w;
    E=A-20*w; F=B-20*w;
    % draw the data
    plot(0,0,'k+'); hold on;
    patch([A(1) B(1) D(1) C(1) A(1)],...
        [A(2) B(2) D(2) C(2) A(2)],[0.7 0.7 0.8]);
    patch([A(1) B(1) F(1) E(1) A(1)],...
        [A(2) B(2) F(2) E(2) A(2)],[0.8 0.7 0.7]);
    plot(X(1,:),X(2,:),'b*');
    plot(Y(1,:),Y(2,:),'r*'); 
    h=plot([0 w(1)],[0 w(2)]);
    set(h,'LineWidth',7,'Marker','.')
    h=plot(b*[0 w(1)],b*[0 w(2)]);
    set(h,'LineWidth',2,'Color','k');
    h=plot([A(1) B(1)],[A(2),B(2)]); hold off;
    set(h,'LineWidth',2,'Color','k');
    axis([-4 10 -4 10]); axis square
    grid on
    title(num2str(n))
    
    %if n<6; pause; end
    drawnow
end



